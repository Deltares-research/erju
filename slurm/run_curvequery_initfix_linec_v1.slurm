#!/bin/bash
#SBATCH --job-name=erju_curvequery_initfix
#SBATCH --output=/u/camposmo/erju/outputs/curvequery_initfix_linec_v1/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/curvequery_initfix_linec_v1/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=================================================================================="
echo "CurveQuery Init-Fix Experiment  —  zero-init intensity/residual final layers"
echo "=================================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Submitted at: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd to repo"; exit 1; }
echo "Working directory: $(pwd)"

echo ""
echo "=== Environment Check ==="
echo "Python: $(which python)"
python --version

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "=== PyTorch Status ==="
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
PY

echo ""
echo "=================================================================================="
echo "EXPERIMENT OVERVIEW"
echo "=================================================================================="
echo "Fix: CurvePriorCNN2D_Query final-layer zero-initialization"
echo "  - intensity_head final layer: weight=0, bias=c_hat_init (train mean c_target)"
echo "  - residual_head final layer:  weight=0, bias=0"
echo "  - c_hat_init computed from training data after n fitting"
echo ""
echo "Expected outcome vs stability run (lr=1e-4):"
echo "  Q1_all:  frac_clip_hi=0 from epoch 1, c_hat starts at ~0.3"
echo "           RMSE(log) ~ 0.61-0.63  (matches P1_corrected_n = 0.6146)"
echo "  Q3_all:  stable convergence, RMSE(log) ~ 0.59-0.60"
echo "  holdout: MP2 excellent, MP4 reported honestly as hard case"
echo ""
echo "lr = 1e-4  |  L_c deduplicated per event  |  same split seed=42"
echo ""
echo "Grid (5 runs):"
echo "  Q1_all_initfix"
echo "  Q2_all_initfix"
echo "  Q3_all_initfix"
echo "  Q3_holdout_MP4_initfix"
echo "  Q3_holdout_MP2_initfix"
echo ""

mkdir -p /u/camposmo/erju/outputs/curvequery_initfix_linec_v1
LOGDIR="/u/camposmo/erju/outputs/curvequery_initfix_linec_v1"
OUTPUT_TAG="cnn_curvequery_initfix_linec_v001"

# ============================================================================
# PHASE 1: ALL-SENSOR VARIANTS
# ============================================================================

echo ""
echo "=========================================================================="
echo "PHASE 1: ALL-SENSOR  (lr=1e-4, init-fix)"
echo "=========================================================================="

echo ""
echo "--- Q1_all_initfix: Curve-only, expected RMSE(log) ~ 0.61-0.63 ---"
echo "Start: $(date)"
python train_cnn_curvequery_linec_v1.py \
  --variant Q1 \
  --lr 1e-4 \
  --output_tag "$OUTPUT_TAG" \
  2>&1 | tee -a "$LOGDIR/Q1_all_initfix.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Q1_all_initfix failed! Continuing..."
else
    echo "Q1_all_initfix complete: $(date)"
fi

echo ""
echo "--- Q2_all_initfix: Curve + residual ---"
echo "Start: $(date)"
python train_cnn_curvequery_linec_v1.py \
  --variant Q2 \
  --lr 1e-4 \
  --output_tag "$OUTPUT_TAG" \
  2>&1 | tee -a "$LOGDIR/Q2_all_initfix.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Q2_all_initfix failed! Continuing..."
else
    echo "Q2_all_initfix complete: $(date)"
fi

echo ""
echo "--- Q3_all_initfix: Curve + residual + MP4 weight, expected RMSE(log) ~ 0.59-0.60 ---"
echo "Start: $(date)"
python train_cnn_curvequery_linec_v1.py \
  --variant Q3 \
  --lr 1e-4 \
  --output_tag "$OUTPUT_TAG" \
  2>&1 | tee -a "$LOGDIR/Q3_all_initfix.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Q3_all_initfix failed! Continuing..."
else
    echo "Q3_all_initfix complete: $(date)"
fi

# ============================================================================
# PHASE 2: HELD-OUT SENSOR VARIANTS
# ============================================================================

echo ""
echo "=========================================================================="
echo "PHASE 2: HELD-OUT SENSORS  (lr=1e-4, init-fix)"
echo "=========================================================================="

echo ""
echo "--- Q3_holdout_MP4_initfix: Close-field extrapolation (hardest case) ---"
echo "Start: $(date)"
python train_cnn_curvequery_linec_v1.py \
  --variant Q3 \
  --holdout_sensor MP4 \
  --lr 1e-4 \
  --output_tag "$OUTPUT_TAG" \
  2>&1 | tee -a "$LOGDIR/Q3_holdout_MP4_initfix.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Q3_holdout_MP4_initfix failed! Continuing..."
else
    echo "Q3_holdout_MP4_initfix complete: $(date)"
fi

echo ""
echo "--- Q3_holdout_MP2_initfix: Far-field extrapolation (best expected case) ---"
echo "Start: $(date)"
python train_cnn_curvequery_linec_v1.py \
  --variant Q3 \
  --holdout_sensor MP2 \
  --lr 1e-4 \
  --output_tag "$OUTPUT_TAG" \
  2>&1 | tee -a "$LOGDIR/Q3_holdout_MP2_initfix.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Q3_holdout_MP2_initfix failed! Continuing..."
else
    echo "Q3_holdout_MP2_initfix complete: $(date)"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=================================================================================="
echo "ALL VARIANTS ATTEMPTED: $(date)"
echo "=================================================================================="
echo ""
echo "Output logs:   $LOGDIR/"
echo "Model outputs: /p/11210978-erju-ai/holten_models/${OUTPUT_TAG}_v*"
echo ""
echo "Key check — Q1_all_initfix first epoch diagnostics:"
echo "  frac_clip_hi should be 0.000 (no clamping)"
echo "  c_hat should start near c_hat_init (~0.3)"
echo "  pred_max should be < 3.0"
echo ""
echo "Decision criteria:"
echo "  Q1 RMSE(log) < 0.65  ->  init fix worked, training stable"
echo "  Q3 RMSE(log) < 0.60  ->  query model matches/beats P3_corrected_n"
echo "  Q3_holdout_MP2 held RMSE(log) < 0.55  ->  far-field generalises well"
echo "  Q3_holdout_MP4 is honest hard case; do not expect < 0.80"
echo ""
