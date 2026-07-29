#!/bin/bash
#SBATCH --job-name=mp8_spec_finals
#SBATCH --array=0-5%4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/u/camposmo/erju/outputs/mp8_spectral_cnn_v001/slurm_finals_%A_%a.log
#SBATCH --error=/u/camposmo/erju/outputs/mp8_spectral_cnn_v001/slurm_finals_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

# ── Finalist configs (update after Stage 1 sweep review) ────────────────────
# Placeholder: S2, S3 — replace with actual winners.
# 2 finalist configs × 3 seeds = 6 array tasks (indices 0–5)
FINALIST_CONFIGS=(S2 S3)
SEEDS=(41 42 43)

N_SEEDS=${#SEEDS[@]}
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

CONFIG=${FINALIST_CONFIGS[$CONFIG_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=========================================================================="
echo "MP8 Spectral CNN PoC — Finalist Seeds"
echo "=========================================================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID / config=${CONFIG} seed=${SEED}"
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

mkdir -p /u/camposmo/erju/outputs/mp8_spectral_cnn_v001

echo ""
echo "=== Training: config=${CONFIG} seed=${SEED} ==="
echo "Start: $(date)"

python train_mp8_spectral_v001.py \
    --config ${CONFIG} \
    --seed ${SEED} \
    --mode full \
    --out_root /p/11210978-erju-ai/holten_models/outputs

EXIT_CODE=$?

echo ""
echo "=== Finished: config=${CONFIG} seed=${SEED}  exit=${EXIT_CODE}  at $(date) ==="
exit ${EXIT_CODE}
