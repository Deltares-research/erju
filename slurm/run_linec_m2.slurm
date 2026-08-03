#!/bin/bash
#SBATCH --job-name=linec_m2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/u/camposmo/erju/outputs/linec_multisensor_v1/M2/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/linec_multisensor_v1/M2/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

# Usage: sbatch slurm/run_linec_m2.slurm <seed> [--smoke]
# e.g.   sbatch slurm/run_linec_m2.slurm 42
#        sbatch slurm/run_linec_m2.slurm 42 --smoke
SEED=$1
shift || true
EXTRA_ARGS="$@"

echo "=========================================================================="
echo "Line-C multi-sensor experiment -- M2 (O0 source-propagation decode), single GPU"
echo "=========================================================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Seed         : $SEED"
echo "Extra args   : $EXTRA_ARGS"
echo "Submitted    : $(date)"
echo "Hostname     : $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# GPU-only, no CPU fallback: fail fast at the shell level before touching Python.
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

mkdir -p /p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1/M2

echo ""
echo "=== Running model=M2 seed=$SEED ==="
echo "Start: $(date)"

python run_linec_multisensor_v1.py \
    --model M2 \
    --seed "$SEED" \
    --num-workers 4 \
    --output-root /p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1/M2 \
    $EXTRA_ARGS

EXIT_CODE=$?
echo ""
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
