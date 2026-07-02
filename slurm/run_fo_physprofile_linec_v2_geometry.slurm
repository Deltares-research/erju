#!/bin/bash
#SBATCH --job-name=erju_fo_geom_v2
#SBATCH --output=/u/camposmo/erju/outputs/fo_physprofile_linec_v2_geometry/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/fo_physprofile_linec_v2_geometry/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=================================================================================="
echo "FO-PhysProfile v2 Geometry — Holten Line-C"
echo "=================================================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Submitted: $(date)"
echo "Host:      $(hostname)"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd"; exit 1; }
echo "Working dir: $(pwd)"
echo ""
echo "=== Environment ==="
python --version
python -c "import xgboost, numpy, scipy, sklearn; print('xgb:', xgboost.__version__, 'np:', numpy.__version__)"

mkdir -p /u/camposmo/erju/outputs/fo_physprofile_linec_v2_geometry

echo ""
echo "=================================================================================="
echo "ARCHITECTURE — FO-PhysProfile v2 Geometry"
echo "=================================================================================="
echo ""
echo "Core addition: Explicit FO-relative coordinate system"
echo ""
echo "Holten coordinate system (from sites/holten.json):"
echo "  FO cable:  Y = 0"
echo "  Track 1:   Y = 4.0 m"
echo "  Track 2:   Y = 8.0 m"
echo "  Line-C side -1 sensors:"
echo "    MP4:  Y = +1.5  fo_axis_pos=0.375  BETWEEN FO and track"
echo "    MP8:  Y =  0.0  fo_axis_pos=0.000  ON FO cable"
echo "    MP10: Y = -4.0  fo_axis_pos=-1.00  BEYOND FO away from track"
echo "    MP1:  Y = -12.0 fo_axis_pos=-3.00  BEYOND FO (far)"
echo "    MP2:  Y = -19.0 fo_axis_pos=-4.75  BEYOND FO (far)"
echo ""
echo "Key new feature: fo_to_track_axis_pos = (sensor_y - fo_y) / (track_y - fo_y)"
echo "  Interpretation:"
echo "    =0  → on FO cable"
echo "    0..1 → between FO and track (MP4 is here)"
echo "    <0   → behind FO, away from track (MP10, MP1, MP2)"
echo "    >1   → beyond track"
echo ""
echo "Variants trained (local residual only, profile reused from v1 pipeline):"
echo "  V1_ref               — v1 feature set (reference)"
echo "  V2_geom              — v1 + geometry + geometry×FO interactions"
echo "  V2_geom_no_sc        — ablation: drop sensor_code"
echo ""
echo "Selection: val RMSE(PGV), tie-break MP4 RMSE"
echo "Grid: depth=[3,4,5] × lr=[0.01,0.02,0.05] × mcw=[1,5] × rl=[1,10,20]"
echo "      schemes=[uniform, agg_hi]"
echo ""
echo "Baselines:"
echo "  P3:                   RMSE(PGV)=2.4006  MP4=5.0185"
echo "  PXGBR-R2:             RMSE(PGV)=2.2718  MP4=4.7389"
echo "  PXGBR_ens_top5:       RMSE(PGV)=2.2473  MP4=4.6680"
echo "  FO-PhysProfile v1:    RMSE(PGV)=2.1882  MP4=4.5592  <- beat this"
echo ""
echo "Success criteria:"
echo "  MP4 RMSE < 4.50"
echo "  Global RMSE(PGV) <= 2.1882"
echo "  RMSE(log) <= 0.60"
echo "=================================================================================="
echo ""

# Verify data + config
echo "=== Pre-run verification ==="
python -c "
from pathlib import Path
import sys

data_root = Path('/p/11210978-erju-ai')
repo_root = Path('/u/camposmo/erju')

# holten.json
hj = repo_root / 'sites' / 'holten.json'
if hj.exists():
    import json
    cfg = json.loads(hj.read_text())
    coords = cfg['accelerometer']['sensor_coordinates_m']
    fo = cfg['fibre_optics']
    print(f'  holten.json: track_1_to_fo_m={fo[\"track_1_to_fo_m\"]}  track_2={fo[\"track_2_to_fo_m\"]}')
    print(f'  MP4 y={coords[\"MP4\"][\"y\"]}  MP8 y={coords[\"MP8\"][\"y\"]}  MP10 y={coords[\"MP10\"][\"y\"]}')
else:
    print('  ERROR: sites/holten.json not found'); sys.exit(1)

# parquet v002
pq = sorted((data_root / 'holten_parquet').glob('parquet_v002_*'))
if pq: print(f'  parquet_v002: {pq[-1].name}')
else: print('  ERROR: No parquet_v002_* found'); sys.exit(1)

# waveform v003
wv = sorted((data_root / 'holten_waveform').glob('holten_waveform_v003_ch51_*'))
if wv:
    import numpy as np
    shape = np.load(wv[-1] / 'waveforms.npy', mmap_mode='r').shape
    print(f'  waveform v003: {wv[-1].name}  shape={shape}')
else:
    print('  WARN: No holten_waveform_v003_ch51_* — will try fallback')

# v1 output (for baseline comparison)
v1_out = sorted((data_root / 'holten_models' / 'outputs' / 'fo_physprofile_linec_v1').glob('fo_physprofile_linec_v001_*'))
if v1_out and (v1_out[-1] / 'profile_targets_event.parquet').exists():
    print(f'  v1 output: {v1_out[-1].name}')
else:
    print('  WARN: No v1 output found — v1 metrics will not appear in comparison')

print('  Verification PASSED')
"

echo ""
echo "=================================================================================="
echo "RUNNING: train_fo_physprofile_linec_v2_geometry.py"
echo "Start: $(date)"
echo "=================================================================================="

python train_fo_physprofile_linec_v2_geometry.py

EXIT_CODE=$?
echo ""
echo "=================================================================================="
echo "FINISHED at $(date)  exit_code=$EXIT_CODE"
echo "=================================================================================="

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: exit $EXIT_CODE"; exit $EXIT_CODE
fi

echo ""
echo "=== Results Summary ==="
python -c "
from pathlib import Path
import pandas as pd

out_root = Path('/p/11210978-erju-ai/holten_models/outputs/fo_physprofile_linec_v2_geometry')
runs = sorted(out_root.glob('fo_physprofile_linec_v002_geom_*'))
if not runs:
    print('No output directory found'); exit()
run = runs[-1]
print(f'Output dir: {run}')

# Metrics table
mf = run / 'metrics_table.csv'
if mf.exists():
    df = pd.read_csv(mf)
    print()
    print('Test metrics (sorted by RMSE PGV):')
    cols = ['model', 'rmse_pgv', 'rmse_log', 'rmse_MP4']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(10).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# Geometry diagnostics
gd = run / 'geometry_diagnostic_table.csv'
if gd.exists():
    print()
    print('Geometry diagnostic (Track 1):')
    df = pd.read_csv(gd)
    t1 = df[df['track'] == 1]
    print(t1[['sensor', 'sensor_y_m', 'fo_to_track_axis_pos',
              'acc_to_active_track_m', 'is_between_track_and_fo',
              'is_on_fo_line', 'is_beyond_fo_away']].to_string(index=False))
"

echo ""
echo "Job complete."
