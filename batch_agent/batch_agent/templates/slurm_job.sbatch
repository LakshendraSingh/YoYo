#!/bin/bash
# ==============================================================================
# Slurm Batch Job Specification for YoYo Autonomous Agent
# 
# DISCLAIMER & SAFETY NOTICE:
# YoYo executes autonomously and can make mistakes. Always run batch jobs
# against copies of your workspace and maintain verified backups of critical data.
# ==============================================================================
#SBATCH --job-name=yoyo_ai
#SBATCH --output=yoyo_%j.out
#SBATCH --error=yoyo_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=batch

set -euo pipefail

echo "========================================================================"
echo "Starting YoYo Agent Job on Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local_run}"
echo "Current Time: $(date -u)"
echo "========================================================================"

# Set non-interactive environment variables
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export CI=true

# Launch the autonomous agent with full sandboxing and audit trail
python3 -m batch_agent.cli \
    --goal "Execute non-interactive pipeline, verify unit tests, and generate audit report" \
    --files "data/input.csv" "src/pipeline.py" \
    --workspace "${SLURM_SUBMIT_DIR:-$(pwd)}" \
    --sandbox-profile strict \
    --log-file "yoyo_audit_${SLURM_JOB_ID:-run}.txt"

EXIT_STATUS=$?
echo "========================================================================"
echo "YoYo Completed with Exit Code: ${EXIT_STATUS}"
echo "Audit Trail written to: yoyo_audit_${SLURM_JOB_ID:-run}.txt"
echo "========================================================================"

exit ${EXIT_STATUS}
