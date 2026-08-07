#!/bin/bash
#SBATCH -J ml_ldmx_baseline_cpu
#SBATCH -p lu48
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH --no-requeue
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

# CPU resource envelope for the shared baseline training implementation.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export REPO_ROOT
export DEVICE=cpu

exec bash "${REPO_ROOT}/scripts/sbatch/cosmos_train_baseline.sbatch"
