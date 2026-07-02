#!/bin/bash
#SBATCH --job-name=erju_fo_physprofile
#SBATCH --output=/u/camposmo/erju/outputs/fo_physprofile_linec_v1/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/fo_physprofile_linec_v1/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=================================================================================="
echo "FO-PhysProfile v1 — Source-path-receiver attenuation model — Holten Line-C"
echo "=================================================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Submitted: $(date)"
echo "Host:      $(hostname)"

# ── Environment ───────────────────────────────────────────────────────────────
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd to erju"; exit 1; }
echo "Working dir: $(pwd)"
echo ""
echo "=== Environment ==="
echo "Python:  $(which python)"
python --version
python -c "
import xgboost, numpy, scipy, sklearn
print('xgboost:', xgboost.__version__)
print('numpy:',   numpy.__version__)
print('scipy:',   scipy.__version__)
print('sklearn:', sklearn.__version__)
"

mkdir -p /u/camposmo/erju/outputs/fo_physprofile_linec_v1

echo ""
echo "=================================================================================="
echo "ARCHITECTURE — FO-PhysProfile v1"
echo "=================================================================================="
echo ""
echo "Physics decomposition:"
echo "  pred_log_ij = c_hat_i"
echo "              - (n_track + delta_n_hat_i) * log(r_ij / r0)"
echo "              + PCA_shape_i[sensor_j]"
echo "              + local_residual_hat_ij"
echo ""
echo "Event-level models (XGBoost on aggregated FO + waveform features):"
echo "  C:   predict c_target_event     (fresh physics target, not P3 c_hat)"
echo "  N:   predict delta_n_target     (per-event n correction, shrunk to global)"
echo "  PC1: predict pc1_target         (residual profile shape PC1)"
echo "  PC2: predict pc2_target         (residual profile shape PC2)"
echo ""
echo "Row-level model (XGBoost local residual):"
echo "  target = target_log - pred_log_profile"
echo "  features: pred_profile, c_hat, n_hat, pc hats, distance, sensor,"
echo "            event FO features, waveform spatial features,"
echo "            distance × frequency interactions"
echo ""
echo "Feature sources:"
echo "  1. parquet_v002 1/3-octave FO features (aggregated event-level: mean/max/std)"
echo "  2. Waveform v003_ch51 spatial + temporal features (21 channels, 250 Hz)"
echo "     - Time-domain: rms, peak, energy, crest_factor, duration_above_threshold"
echo "     - Frequency: band energies 1-5/5-10/10-20/20-40/40-80/80-125 Hz"
echo "     - Spatial: ch_energy stats, centroid, spread, asymmetry, adj_correlation"
echo "     - Pre/post: pre-event rms, post-pre contrast, cumulative strain"
echo ""
echo "Physics targets:"
echo "  c_target = mean(target_log + n_track * log(r/r0)) over 5 sensors"
echo "  n_event_raw: per-event OLS fit on 5-sensor profile"
echo "  n_event_shrunk = 0.7*n_track + 0.3*n_event_raw  (clipped [0.5, 2.0])"
echo "  PCA fitted on TRAINING residual profiles only (n_comp=3)"
echo ""
echo "Corrected n values (from event-intercept method):"
echo "  track 1: n = 1.0655"
echo "  track 2: n = 1.3246"
echo "  r0 = 10 m"
echo ""
echo "Baselines to beat:"
echo "  P3_corrected_n:  RMSE(PGV)=2.4006"
echo "  PXGBR-R2:        RMSE(PGV)=2.2718"
echo "  PXGBR_ens_top5:  RMSE(PGV)=2.2473  <- primary target"
echo "  A1_direct:       RMSE(PGV)=2.2445"
echo ""
echo "Targets:"
echo "  RMSE(PGV) < 2.20"
echo "  RMSE(log) <= 0.61"
echo "  MP4 RMSE  < 4.50"
echo "  mono_viol_rate near zero"
echo ""
echo "Success criterion:"
echo "  c_hat vs c_target R² > 0.30 on val"
echo "=================================================================================="
echo ""

# ── Verify data paths ─────────────────────────────────────────────────────────
echo "=== Pre-run data verification ==="
python -c "
from pathlib import Path
import sys

data_root = Path('/p/11210978-erju-ai')

# parquet_v002
pq = sorted((data_root / 'holten_parquet').glob('parquet_v002_*'))
if pq:
    print(f'  parquet_v002: {pq[-1].name}')
else:
    print('  ERROR: No parquet_v002_* found!'); sys.exit(1)

# waveform v003 ch51
wv = sorted((data_root / 'holten_waveform').glob('holten_waveform_v003_ch51_*'))
if wv:
    import numpy as np
    shape = np.load(wv[-1] / 'waveforms.npy', mmap_mode='r').shape
    print(f'  waveform_v003_ch51: {wv[-1].name}  shape={shape}')
else:
    print('  WARN: No holten_waveform_v003_ch51_* found — will try fallback')

# P3 savepred
p3 = sorted((data_root / 'holten_models').glob('cnn_curveprior_p3_savepred_linec_v001_vP3_*'))
if p3:
    print(f'  P3 savepred: {p3[-1].name}')
else:
    print('  WARN: No P3 savepred found — split will be derived from data')

print('  Data verification PASSED')
"

echo ""
echo "=================================================================================="
echo "RUNNING: train_fo_physprofile_linec_v1.py"
echo "Start time: $(date)"
echo "=================================================================================="

# ── Run the training script ────────────────────────────────────────────────────
python train_fo_physprofile_linec_v1.py

EXIT_CODE=$?
echo ""
echo "=================================================================================="
echo "FINISHED at $(date)  exit_code=$EXIT_CODE"
echo "=================================================================================="

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training script exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=== Summary of outputs ==="
python -c "
from pathlib import Path
import pandas as pd

out_root = Path('/p/11210978-erju-ai/holten_models/outputs/fo_physprofile_linec_v1')
runs = sorted(out_root.glob('fo_physprofile_linec_v001_*'))
if runs:
    run = runs[-1]
    print(f'Output dir: {run}')
    metrics_path = run / 'metrics_table.csv'
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        print()
        print('Metrics table (sorted by RMSE PGV):')
        print(df[['model', 'rmse_pgv', 'rmse_log', 'r2_log']].to_string(index=False))
    phys_path = run / 'physics_diagnostics.json'
    if phys_path.exists():
        import json
        diag = json.loads(phys_path.read_text())
        print()
        print('Physics diagnostics:')
        print(f'  c_hat R2 (val):      {diag.get(\"c_hat_r2_val\", \"?\"):.4f}')
        print(f'  n_hat R2 (val):      {diag.get(\"n_hat_r2_val\", \"?\"):.4f}')
        print(f'  PC1 R2 (val):        {diag.get(\"pc1_r2_val\", \"?\"):.4f}')
        print(f'  PC2 R2 (val):        {diag.get(\"pc2_r2_val\", \"?\"):.4f}')
        print(f'  PCA VE:              {diag.get(\"pca_ve\", \"?\")}')
        print(f'  Profile RMSE PGV val:{diag.get(\"profile_rmse_pgv_val\", \"?\"):.4f}')
        print(f'  WF features:         {diag.get(\"wf_features_ok\", \"?\")}')
else:
    print('No output directory found')
"

echo ""
echo "Job complete."
