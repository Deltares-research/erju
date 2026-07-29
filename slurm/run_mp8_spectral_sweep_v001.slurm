#!/bin/bash
#SBATCH --job-name=mp8_spec_sweep
#SBATCH --array=0-5%4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/u/camposmo/erju/outputs/mp8_spectral_cnn_v001/slurm_sweep_%A_%a.log
#SBATCH --error=/u/camposmo/erju/outputs/mp8_spectral_cnn_v001/slurm_sweep_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

# ── Config array (6 experiment configurations) ──────────────────────────────
CONFIGS=(S1 S2 S3 S4 S5 P1)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================================================="
echo "MP8 Spectral CNN PoC — Initial Sweep"
echo "=========================================================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID / config=${CONFIG}"
echo "Submitted : $(date)"
echo "Hostname  : $(hostname)"

# ── Environment setup ────────────────────────────────────────────────────────
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd to /u/camposmo/erju"; exit 1; }
echo "Working dir: $(pwd)"

echo ""
echo "=== Environment ==="
echo "Python  : $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "=== PyTorch ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Ensure output dir exists
mkdir -p /u/camposmo/erju/outputs/mp8_spectral_cnn_v001

echo ""
echo "=== Training: config=${CONFIG} seed=42 ==="
echo "Start: $(date)"

python train_mp8_spectral_v001.py \
    --config ${CONFIG} \
    --seed 42 \
    --mode full \
    --out_root /p/11210978-erju-ai/holten_models/outputs

EXIT_CODE=$?

echo ""
echo "=== Finished: config=${CONFIG}  exit=${EXIT_CODE}  at $(date) ==="
exit ${EXIT_CODE}
