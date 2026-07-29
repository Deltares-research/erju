#!/bin/bash
#SBATCH --job-name=mp8_spec_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/u/camposmo/erju/outputs/mp8_spectral_cnn_v001/slurm_smoke_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/mp8_spectral_cnn_v001/slurm_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=========================================================================="
echo "MP8 Spectral CNN PoC — Stage 0 Smoke Test"
echo "=========================================================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Submitted : $(date)"
echo "Hostname  : $(hostname)"

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
echo "=== Smoke Test ==="
echo "Start: $(date)"

python smoke_test_spectral_v001.py

EXIT_CODE=$?
echo ""
echo "=== Finished  exit=${EXIT_CODE}  at $(date) ==="
exit ${EXIT_CODE}
