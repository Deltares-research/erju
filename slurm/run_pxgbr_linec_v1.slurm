#!/bin/bash
#SBATCH --job-name=erju_pxgbr_linec
#SBATCH --output=/u/camposmo/erju/outputs/pxgbr_linec_v1/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/pxgbr_linec_v1/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=================================================================================="
echo "PXGB-R: Physics-Boosted Residual XGBoost  —  Holten Line-C side -1"
echo "=================================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Submitted at: $(date)"
echo "Hostname: $(hostname)"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd to repo"; exit 1; }
echo "Working directory: $(pwd)"

echo ""
echo "=== Environment ==="
echo "Python: $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "=== PyTorch + XGBoost ==="
python - <<'PY'
import torch, xgboost
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("xgboost:", xgboost.__version__)
PY

mkdir -p /u/camposmo/erju/outputs/pxgbr_linec_v1
LOGDIR="/u/camposmo/erju/outputs/pxgbr_linec_v1"

echo ""
echo "=================================================================================="
echo "EXPERIMENT OVERVIEW"
echo "=================================================================================="
echo "Architecture:  pred_log_final = pred_log_p3 + residual_xgb"
echo "               where residual_xgb trained on (target_log - pred_log_p3)"
echo ""
echo "Variants:"
echo "  PXGBR-R1: uniform weights"
echo "  PXGBR-R2: MP4 +1.5 weight, high-PGV (>4/8 mm/s) +1.0 weight"
echo ""
echo "Physics prior: P3_corrected_n (RMSE(log)=0.5993, RMSE(PGV)=2.401)"
echo ""
echo "Expected improvement targets:"
echo "  Conservative: RMSE(PGV) < 2.20 mm/s (8% improvement over P3)"
echo "  Strong:       RMSE(PGV) ~ 1.9-2.1 mm/s"
echo ""
echo "Feature set: P3 physics predictions + FO octave bands + train metadata"
echo ""

# ============================================================================
# STEP 1: Run P3 savepred (saves all_predictions.parquet needed by PXGB-R)
# ============================================================================

echo ""
echo "=========================================================================="
echo "STEP 1: P3 savepred — rerun to save all-splits predictions"
echo "=========================================================================="
echo "Start: $(date)"
echo "This saves: model.pth, {train,val,test,all}_predictions.parquet"

python train_cnn_curveprior_linec_v1_savepred.py \
  --variant P3 \
  --n_mode fit_corrected \
  --lambda_residual 0.05 \
  --epochs 100 \
  2>&1 | tee -a "$LOGDIR/p3_savepred.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] P3 savepred failed! Cannot proceed without all_predictions.parquet"
    exit 1
fi
echo "P3 savepred complete: $(date)"

# ============================================================================
# STEP 2: Train PXGB-R
# ============================================================================

echo ""
echo "=========================================================================="
echo "STEP 2: Train PXGB-R (R1 uniform + R2 weighted)"
echo "=========================================================================="
echo "Start: $(date)"

python train_pxgbr_linec_v1.py \
  2>&1 | tee -a "$LOGDIR/pxgbr_train.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] PXGBR training failed!"
    exit 1
fi
echo "PXGBR training complete: $(date)"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=================================================================================="
echo "SUCCESS: $(date)"
echo "=================================================================================="
echo ""
echo "Output logs:   $LOGDIR/"
echo "Model outputs: /p/11210978-erju-ai/holten_models/pxgbr_linec_v001_*"
echo ""
echo "Key files:"
echo "  metrics_table.csv    (P3 vs PXGBR-R1 vs R2 vs R2_mono)"
echo "  predictions_test.parquet"
echo "  feature_importance.csv"
echo "  attenuation_profiles_high_pgv_p3_vs_pxgbr.png"
echo ""
echo "Decision criteria:"
echo "  PXGBR-R2 RMSE(PGV) < 2.20  ->  meaningful improvement over P3"
echo "  PXGBR-R2 MP4 RMSE < P3 MP4 ->  high-PGV underprediction reduced"
echo "  Feature importance top-5 should include P3 physics features"
echo ""
