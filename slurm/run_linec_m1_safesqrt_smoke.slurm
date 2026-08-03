#!/bin/bash
#SBATCH --job-name=linec_m1_safesqrt_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/u/camposmo/erju/outputs/linec_multisensor_v1/M1_safesqrt_smoke/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/linec_multisensor_v1/M1_safesqrt_smoke/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

# 5-epoch, full-data smoke run validating the SAFE_SQRT_EPS fix for M1.
# Fresh output directory -- does not reuse any prior M1 checkpoint/output.
# Usage: sbatch slurm/run_linec_m1_safesqrt_smoke.slurm <seed>
# e.g.   sbatch slurm/run_linec_m1_safesqrt_smoke.slurm 42
SEED=$1

echo "=========================================================================="
echo "Line-C M1 -- SAFE_SQRT_EPS fix validation, 5-epoch full-data smoke run"
echo "=========================================================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Seed         : $SEED"
echo "Submitted    : $(date)"
echo "Hostname     : $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

if ! nvidia-smi > /dev/null 2>&1; then
    echo "[FATAL] nvidia-smi found no GPU on this node -- this job must never fall back to CPU. Aborting."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd to /u/camposmo/erju"; exit 1; }
echo "Working dir  : $(pwd)"
echo "Python       : $(which python)"
python --version

OUT_ROOT=/p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1/M1_safesqrt_smoke
mkdir -p "$OUT_ROOT"

echo ""
echo "=== Running model=M1 seed=$SEED epochs=5 (SAFE_SQRT_EPS fix, full data) ==="
echo "Start: $(date)"

python -u run_linec_multisensor_v1.py \
    --model M1 \
    --seed "$SEED" \
    --epochs 5 \
    --patience 5 \
    --num-workers 4 \
    --output-root "$OUT_ROOT"

EXIT_CODE=$?
echo ""
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
