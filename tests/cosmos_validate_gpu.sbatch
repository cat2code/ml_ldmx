#!/bin/bash
#SBATCH -J ml_ldmx_validate_gpu
#SBATCH -p gpua100i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-data/processed/production_10M_001_sharded}"
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-10}"

module --force purge
module load GCCcore/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}"

# Useful for outputs produced by the script itself.
mkdir -p outputs/slurm

source "${VENV_DIR:-${REPO_ROOT}/.venv}/bin/activate"
python -m pip install -e .

echo "host: $(hostname)"
echo "repo: ${REPO_ROOT}"
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
echo "events per source: ${EVENTS_PER_SOURCE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python --version
nvidia-smi

python -u tests/check_cuda_environment.py

ML_LDMX_PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT}" \
ML_LDMX_EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE}" \
ML_LDMX_SMOKE_DEVICE=cuda \
python -m pytest -q tests/test_model_family_smoke.py
