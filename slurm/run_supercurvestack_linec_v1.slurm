#!/bin/bash
#SBATCH --job-name=erju_superstack
#SBATCH --output=/u/camposmo/erju/outputs/supercurvestack_linec_v1/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/supercurvestack_linec_v1/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=================================================================================="
echo "SuperCurveStack v1 — Multimodal Stacked Ensemble — Holten Line-C"
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
python -c "import xgboost,torch; print('xgboost:', xgboost.__version__, '| torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

mkdir -p /u/camposmo/erju/outputs/supercurvestack_linec_v1
LOGDIR="/u/camposmo/erju/outputs/supercurvestack_linec_v1"

echo ""
echo "=================================================================================="
echo "ARCHITECTURE"
echo "=================================================================================="
echo ""
echo "Base A: Direct log-PGV XGBoost"
echo "  A1: no physics features  (FO + metadata + geometry)"
echo "  A2: with physics features (+ pred_log_p3, epsilon_p3, n_used)"
echo "  Grid: depth=[3,4,5] x lr=[0.02,0.05] x mcw=[1,5] x rl=[1,10] x scheme=[uniform,D]"
echo "  = 96 fits"
echo ""
echo "Base B: Physics residual XGBoost (focused grid around best ensemble config)"
echo "  target = target_log - pred_log_p3"
echo "  Grid: depth=[3,4] x lr=[0.02,0.05] x mcw=[1,5] x rl=[5,10] x ss=[0.9] x cbt=[0.8,0.9]"
echo "  Schemes: uniform, A, D"
echo "  = 48 fits + top-5 ensemble"
echo ""
echo "Base C: PGV-scaled residual XGBoost"
echo "  target = (target_pgv - pred_pgv_p3) / sqrt(pred_pgv_p3 + 1)"
echo "  = 16 fits"
echo ""
echo "Base D: Event-level c/n attenuation model (fresh tabular c_hat)"
echo "  Aggregate FO features over 5 sensors, predict c_target_event"
echo "  Reconstruct: pred_log = c_hat - n_track * log(r/r0)"
echo "  = 12 fits"
echo ""
echo "Base E: Small tabular MLP (3-layer, residual target)"
echo "  3 random seeds"
echo ""
echo "Meta-learner (trained on val base predictions):"
echo "  1. Non-negative linear blend (scipy minimize)"
echo "  2. Ridge regression"
echo "  3. Shallow XGBoost (depth 2-3)"
echo "  Selection: val RMSE(PGV)"
echo ""
echo "Baselines:"
echo "  P3_corrected_n:  RMSE(PGV)=2.4006"
echo "  PXGBR-R2:        RMSE(PGV)=2.2718"
echo "  PXGBR_ens_top5:  RMSE(PGV)=2.2473   <- beat this"
echo ""
echo "Targets:"
echo "  Beat PXGBR ensemble: RMSE(PGV) < 2.2473"
echo "  Target:              RMSE(PGV) < 2.20"
echo "  Stretch:             RMSE(PGV) < 2.00"
echo "  MP4 target:          RMSE(PGV) < 4.50"
echo ""

# ============================================================================
# STEP 1: Ensure P3 savepred exists
# ============================================================================

P3_PRED=$(ls -d /p/11210978-erju-ai/holten_models/cnn_curveprior_p3_savepred_linec_v001_vP3_* 2>/dev/null | sort | tail -1)

if [ -n "$P3_PRED" ] && [ -f "$P3_PRED/all_predictions.parquet" ]; then
    echo "Found existing P3 savepred: $P3_PRED"
else
    echo "Running P3 savepred..."
    python train_cnn_curveprior_linec_v1_savepred.py \
      --variant P3 --n_mode fit_corrected --lambda_residual 0.05 --epochs 100 \
      2>&1 | tee -a "$LOGDIR/p3_savepred.log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] P3 savepred failed! Exiting."
        exit 1
    fi
fi

# ============================================================================
# STEP 2: Train SuperCurveStack
# ============================================================================

echo ""
echo "=========================================================================="
echo "STEP 2: SuperCurveStack training"
echo "=========================================================================="
echo "Start: $(date)"

python train_supercurvestack_linec_v1.py \
  2>&1 | tee -a "$LOGDIR/superstack_train.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] SuperCurveStack training failed!"
    exit 1
fi
echo "SuperCurveStack complete: $(date)"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=================================================================================="
echo "SUCCESS: $(date)"
echo "=================================================================================="
echo ""
echo "Output: /p/11210978-erju-ai/holten_models/supercurvestack_linec_v001_*"
echo ""
echo "Key files:"
echo "  metrics_table.csv          — all base models + meta-learner"
echo "  predictions_test.parquet   — all variants, test split"
echo "  base_predictions_val.parquet"
echo "  meta_weights.json"
echo "  feature_importance_*.csv"
echo ""
echo "Decision criteria:"
echo "  meta_best RMSE(PGV) < 2.2473  ->  beats PXGBR ensemble"
echo "  meta_best RMSE(PGV) < 2.20    ->  target achieved"
echo "  MP4 RMSE < 4.50               ->  close-field improvement"
echo ""
