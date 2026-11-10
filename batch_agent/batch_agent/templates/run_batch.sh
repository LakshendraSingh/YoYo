#!/usr/bin/env bash
# ==============================================================================
# Portable Batch Runner for YoYo Autonomous Agent
# Runs in cron, systemd background units, Jenkins, or headless CI/CD
#
# DISCLAIMER & SAFETY NOTICE:
# YoYo executes autonomously and can make mistakes. Always run on dedicated
# working copies and maintain independent backups of all critical code and data.
# ==============================================================================

set -euo pipefail

WORKSPACE="${1:-$(pwd)}"
GOAL="${2:-Autonomously analyze codebase, run verification tests, and produce audit report}"
LOG_FILE="${3:-${WORKSPACE}/yoyo_audit.txt}"

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export CI=true

echo "[$(date -u +%FT%TZ)] Launching YoYo..."

python3 -m batch_agent.cli \
    --goal "${GOAL}" \
    --workspace "${WORKSPACE}" \
    --sandbox-profile workspace \
    --log-file "${LOG_FILE}"

EXIT_CODE=$?
echo "[$(date -u +%FT%TZ)] YoYo exited with status: ${EXIT_CODE}"
exit ${EXIT_CODE}
