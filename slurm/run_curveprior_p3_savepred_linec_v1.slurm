#!/bin/bash
#SBATCH --job-name=erju_p3_savepred
#SBATCH --output=/u/camposmo/erju/outputs/curveprior_p3_savepred_linec_v1/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/curveprior_p3_savepred_linec_v1/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=================================================================================="
echo "P3_corrected_n  —  SAVEPRED rerun (saves model.pth + predictions.parquet)"
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
echo "=== PyTorch ==="
python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
PY

mkdir -p /u/camposmo/erju/outputs/curveprior_p3_savepred_linec_v1

LOGDIR="/u/camposmo/erju/outputs/curveprior_p3_savepred_linec_v1"

echo ""
echo "================================================================================"
echo "Running P3_corrected_n savepred"
echo "Expected: RMSE(log) ~ 0.5993, saves predictions.parquet + model.pth"
echo "================================================================================"
echo "Start: $(date)"

python train_cnn_curveprior_linec_v1_savepred.py \
  --variant P3 \
  --n_mode fit_corrected \
  --lambda_residual 0.05 \
  --epochs 100 \
  2>&1 | tee -a "$LOGDIR/P3_corrected_savepred.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] P3_corrected_savepred failed!"
    exit 1
fi
echo "P3_corrected_savepred complete: $(date)"

echo ""
echo "================================================================================"
echo "SUCCESS: $(date)"
echo "Output: /p/11210978-erju-ai/holten_models/cnn_curveprior_p3_savepred_linec_v001_*"
echo "Files saved: model.pth  predictions.parquet  per_sensor_metrics.csv  metrics.json"
echo "================================================================================"
