#!/bin/bash
#SBATCH --job-name=linec_multisensor_v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=/u/camposmo/erju/outputs/linec_multisensor_v1/slurm_%A_%a.log
#SBATCH --error=/u/camposmo/erju/outputs/linec_multisensor_v1/slurm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

# 3 models x 3 seeds = 9 array tasks. NOT submitted automatically -- prepared only.
echo "=========================================================================="
echo "Line-C multi-sensor experiment (M0/M1/M2, 3 seeds each) -- full cluster run"
echo "=========================================================================="
echo "Job ID       : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Submitted    : $(date)"
echo "Hostname     : $(hostname)"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate erju-torch
cd /u/camposmo/erju || { echo "Failed to cd to /u/camposmo/erju"; exit 1; }
echo "Working dir  : $(pwd)"
echo "Python       : $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

MODELS=(M0 M1 M2)
SEEDS=(42 43 44)
N_SEEDS=${#SEEDS[@]}
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

echo ""
echo "=== Running model=$MODEL seed=$SEED ==="
echo "Start: $(date)"

mkdir -p /p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1

python run_linec_multisensor_v1.py \
    --model "$MODEL" \
    --seed "$SEED" \
    --num-workers 4 \
    --output-root /p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1

EXIT_CODE=$?
echo ""
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
