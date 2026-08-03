#!/bin/bash
#SBATCH --job-name=linec_m2_debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/u/camposmo/erju/outputs/linec_multisensor_v1/M2_debug/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/linec_multisensor_v1/M2_debug/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

# Isolated M2 diagnostic ONLY: replays epoch 1 (full) + epoch 2 batches 0..21
# with the real training loop, then instruments batch 22 (5 separate
# backward passes + anomaly detection) and stops. No full training, no
# optimizer step past batch 22, and no changes to production code.
#
# Usage: sbatch slurm/run_linec_m2_debug.slurm 42
SEED=${1:-42}

echo "=========================================================================="
echo "Line-C M2 debug -- isolated epoch2/batch22 non-finite-gradient diagnostic"
echo "=========================================================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Seed         : $SEED"
echo "Submitted    : $(date)"
echo "Hostname     : $(hostname)"

if ! nvidia-smi > /dev/null 2>&1; then
    echo "[FATAL] nvidia-smi found no GPU on this node -- aborting."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd to /u/camposmo/erju"; exit 1; }
echo "Working dir  : $(pwd)"
python --version

mkdir -p /p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1/M2_debug

echo ""
echo "Start: $(date)"
python -u debug_m2_epoch2_batch22.py \
    --seed "$SEED" \
    --num-workers 0 \
    --output-root /p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1/M2_debug

EXIT_CODE=$?
echo ""
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
