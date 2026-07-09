#!/bin/bash
#SBATCH -J ml_ldmx_slot
#SBATCH -p gpua100i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-data/processed/production_5M_001_sharded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cosmos_slot}"
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-500}"  # total events = 2 * this for 2e+3e
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-7}"
SUPERVISE_NOISE="${SUPERVISE_NOISE:-0}"
RUN_NAME="${RUN_NAME:-slot_${EVENTS_PER_SOURCE}per_source_${SLURM_JOB_ID:-manual}}"
RESUME="${RESUME:-}"
ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM:-log1p}"
TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM:-log1p}"

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}"
mkdir -p outputs/slurm "${OUTPUT_ROOT}"
source "${VENV_DIR:-${REPO_ROOT}/.venv}/bin/activate"
python -m pip install -e .

echo "host: $(hostname)"
echo "repo: ${REPO_ROOT}"
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
echo "events per source: ${EVENTS_PER_SOURCE}"
echo "supervise noise: ${SUPERVISE_NOISE}"
echo "ECal energy transform: ${ECAL_ENERGY_TRANSFORM}"
echo "TPAD PE transform: ${TPAD_PE_TRANSFORM}"
python --version
nvidia-smi
python -u scripts/check_cuda_environment.py

cmd=(
  python -u scripts/train_ecal_tpad_slot_model.py
  --processed-cache-root "${PROCESSED_CACHE_ROOT}"
  --events-per-source "${EVENTS_PER_SOURCE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --dropout "${DROPOUT}"
  --seed "${SEED}"
  --ecal-energy-transform "${ECAL_ENERGY_TRANSFORM}"
  --tpad-pe-transform "${TPAD_PE_TRANSFORM}"
  --device cuda
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --no-progress
)
if [[ "${SUPERVISE_NOISE}" == "1" ]]; then
  cmd+=(--supervise-noise)
fi
if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

printf 'command:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
