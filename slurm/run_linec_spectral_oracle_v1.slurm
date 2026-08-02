#!/bin/bash
#SBATCH --job-name=linec_spec_oracle
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/u/camposmo/erju/outputs/linec_spectral_oracle_v1/slurm_%j.log
#SBATCH --error=/u/camposmo/erju/outputs/linec_spectral_oracle_v1/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.campos@deltares.nl

echo "=========================================================================="
echo "Line-C target-only spectral oracle diagnostic (O0 vs O1) -- full run"
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

mkdir -p /u/camposmo/erju/outputs/linec_spectral_oracle_v1

echo ""
echo "=== Full diagnostic (no --smoke) ==="
echo "Start: $(date)"

python analyse_linec_spectral_oracle_v1.py

EXIT_CODE=$?
echo ""
echo "=== Finished  exit=${EXIT_CODE}  at $(date) ==="
exit ${EXIT_CODE}
