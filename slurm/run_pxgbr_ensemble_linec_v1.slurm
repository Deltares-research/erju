#!/bin/bash
#SBATCH --job-name=erju_pxgbr_ens
#SBATCH --output=/u/camposmo/erju/outputs/pxgbr_ensemble_linec_v1/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/pxgbr_ensemble_linec_v1/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=================================================================================="
echo "PXGBR Ensemble/Loss Sweep  —  Holten Line-C side -1"
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
python -c "import xgboost; print('xgboost:', xgboost.__version__)"

mkdir -p /u/camposmo/erju/outputs/pxgbr_ensemble_linec_v1
LOGDIR="/u/camposmo/erju/outputs/pxgbr_ensemble_linec_v1"

echo ""
echo "=================================================================================="
echo "EXPERIMENT OVERVIEW"
echo "=================================================================================="
echo "Architecture:  pred_log_final = pred_log_p3 + residual_xgb"
echo ""
echo "Grid:"
echo "  max_depth:       [3, 4, 5]"
echo "  learning_rate:   [0.02, 0.05]"
echo "  min_child_weight:[1, 5]"
echo "  reg_lambda:      [1, 10]"
echo "  subsample:       [0.8, 0.9]"
echo "  colsample_bytree:[0.8, 0.9]"
echo "  = 64 hyperparameter combinations"
echo ""
echo "Weight schemes:  uniform, A (R2-base), B (stronger MP4), C (sqrt-PGV), D (aggressive)"
echo "Target variants: log-residual (all 5 schemes) + PGV-scaled (uniform)"
echo "Total fits:      64*6 = 384"
echo ""
echo "Ensembles:  top-3 mean, top-5 mean, top-5 median, inverse-RMSE-weighted"
echo "Selection:  validation RMSE(PGV)  [not residual RMSE]"
echo ""
echo "Baselines:"
echo "  P3_corrected_n:  RMSE(PGV)=2.4006  RMSE(log)=0.5993"
echo "  PXGBR-R2:        RMSE(PGV)=2.2718  RMSE(log)=0.5946"
echo ""
echo "Targets:"
echo "  Beat PXGBR-R2:  RMSE(PGV) < 2.2718"
echo "  Target:         RMSE(PGV) < 2.20"
echo "  MP4 target:     RMSE(PGV) < 4.70"
echo ""

python train_pxgbr_ensemble_linec_v1.py \
  2>&1 | tee -a "$LOGDIR/pxgbr_ensemble.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] PXGBR ensemble training failed!"
    exit 1
fi
echo "PXGBR ensemble complete: $(date)"

echo ""
echo "=================================================================================="
echo "SUCCESS: $(date)"
echo "=================================================================================="
echo ""
echo "Outputs: /p/11210978-erju-ai/holten_models/pxgbr_ensemble_linec_v001_*"
echo ""
echo "Key files:"
echo "  metrics_table.csv"
echo "  ensemble_predictions_test.parquet"
echo "  feature_importance_top30.csv"
echo "  measured_vs_predicted_P3_vs_PXGBR_vs_ensemble.png"
echo "  per_sensor_rmse_P3_vs_PXGBR_vs_ensemble.png"
echo ""
echo "Decision: best_ens RMSE(PGV) vs PXGBR-R2 (2.2718)"
echo ""
