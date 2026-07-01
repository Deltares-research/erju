#!/bin/bash
#SBATCH --job-name=erju_eac_pxgbr
#SBATCH --output=/u/camposmo/erju/outputs/eac_pxgbr_linec_v1/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/eac_pxgbr_linec_v1/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=================================================================================="
echo "EAC-PXGB: Event-Amplitude Calibrated Physics-Boosted XGBoost"
echo "=================================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Submitted at: $(date)"
echo "Hostname: $(hostname)"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd"; exit 1; }
echo "Working directory: $(pwd)"

echo ""
echo "=== Environment ==="
echo "Python: $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "=== XGBoost ==="
python -c "import xgboost; print('xgboost:', xgboost.__version__)"

mkdir -p /u/camposmo/erju/outputs/eac_pxgbr_linec_v1
LOGDIR="/u/camposmo/erju/outputs/eac_pxgbr_linec_v1"

echo ""
echo "=================================================================================="
echo "ARCHITECTURE"
echo "=================================================================================="
echo "Stage 0:  Event high-PGV gate     (XGBoost classifier, target=max_pgv>8)"
echo "Stage 1:  Event Δc correction     (XGBoost regressor, target=c_target-c_hat_p3)"
echo "Stage 2:  Row local residual      (XGBoost regressor, target=target-pred_eac_c)"
echo ""
echo "Final:  pred_log = pred_log_p3 + delta_c_hat + local_residual_hat"
echo ""
echo "Event features (~72 FO octave bands × 4 aggregations + metadata + P3 diagnostics)"
echo "Row features  (row-level FO + physics + calibrated predictions)"
echo ""
echo "Variants:"
echo "  EAC-C1:          Stage 1 uniform event weights"
echo "  EAC-C2:          Stage 1 high-PGV/MP4 up-weighted events (×3)"
echo "  EAC-C2+R1:       Stage 2 uniform row weights"
echo "  EAC-C2+R2:       Stage 2 MP4/high-PGV row weights"
echo "  EAC-C2+R2_mono:  Final monotonic post-processing"
echo ""
echo "Baselines:"
echo "  P3_corrected_n:  RMSE(PGV)=2.4006  RMSE(log)=0.5993"
echo "  PXGBR-R2:        RMSE(PGV)=2.2718  RMSE(log)=0.5946"
echo ""
echo "Targets:"
echo "  Conservative:  RMSE(PGV) < 2.20  (beat PXGBR-R2)"
echo "  Strong:        RMSE(PGV) < 2.10"
echo "  Stretch:       RMSE(PGV) < 2.00"
echo "  MP4 target:    RMSE(PGV) < 4.70"
echo ""

# ============================================================================
# STEP 1: Run P3 savepred (ensures all_predictions.parquet exists)
# ============================================================================

echo ""
echo "=========================================================================="
echo "STEP 1: P3 savepred — ensure all-splits predictions exist"
echo "=========================================================================="
echo "Start: $(date)"

# Check if all_predictions.parquet already exists from a recent run
P3_PRED=$(ls -d /p/11210978-erju-ai/holten_models/cnn_curveprior_p3_savepred_linec_v001_vP3_* 2>/dev/null | sort | tail -1)

if [ -n "$P3_PRED" ] && [ -f "$P3_PRED/all_predictions.parquet" ]; then
    echo "Found existing P3 savepred: $P3_PRED"
    echo "Skipping P3 savepred rerun."
else
    echo "No P3 savepred found — running now..."
    python train_cnn_curveprior_linec_v1_savepred.py \
      --variant P3 \
      --n_mode fit_corrected \
      --lambda_residual 0.05 \
      --epochs 100 \
      2>&1 | tee -a "$LOGDIR/p3_savepred.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] P3 savepred failed! Cannot proceed."
        exit 1
    fi
    echo "P3 savepred complete: $(date)"
fi

# ============================================================================
# STEP 2: Train EAC-PXGB
# ============================================================================

echo ""
echo "=========================================================================="
echo "STEP 2: EAC-PXGB training (Stage 0 + Stage 1 + Stage 2)"
echo "=========================================================================="
echo "Start: $(date)"

python train_eac_pxgbr_linec_v1.py \
  2>&1 | tee -a "$LOGDIR/eac_train.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] EAC-PXGB training failed!"
    exit 1
fi
echo "EAC-PXGB complete: $(date)"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=================================================================================="
echo "SUCCESS: $(date)"
echo "=================================================================================="
echo ""
echo "Output logs:   $LOGDIR/"
echo "Model outputs: /p/11210978-erju-ai/holten_models/eac_pxgbr_linec_v001_*"
echo ""
echo "Key files per run:"
echo "  metrics_table.csv           — all model comparison"
echo "  predictions_test.parquet    — all variants, test split"
echo "  event_predictions_test.parquet — event-level Δc diagnostics"
echo "  high_pgv_gate_metrics.json  — Stage 0 AUC/AP"
echo "  feature_importance_delta_c_C2.csv"
echo "  feature_importance_local_residual_R2.csv"
echo ""
echo "Decision criteria:"
echo "  EAC-C2+R2 RMSE(PGV) < 2.20  ->  beats PXGBR-R2"
echo "  EAC-C2+R2 MP4 RMSE < 4.70   ->  close-field improvement"
echo "  delta_c RMSE small           ->  Stage 1 adds value"
echo "  Stage 0 AUC > 0.75           ->  high-PGV gate useful"
echo ""
