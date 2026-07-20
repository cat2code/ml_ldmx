#!/bin/bash
#SBATCH -J ml_ldmx_baseline
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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cosmos_baselines}"
MODEL="${MODEL:-ECalTpadTransformer}"
SOURCE_LABEL="${SOURCE_LABEL:-}"              # leave empty for balanced 2e+3e; set to 2e or 3e for one source
ELECTRON_COUNT="${ELECTRON_COUNT:-}"          # inferred from SOURCE_LABEL when omitted
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-500}" # total events = sources * this value
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-128}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-7}"
RESUME="${RESUME:-}"
ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM:-log1p}"
TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM:-log1p}"

if [[ -n "${SOURCE_LABEL}" ]]; then
  case "${SOURCE_LABEL}" in
    2e)
      ELECTRON_COUNT="${ELECTRON_COUNT:-2}"
      ;;
    3e)
      ELECTRON_COUNT="${ELECTRON_COUNT:-3}"
      ;;
    *)
      echo "SOURCE_LABEL must be empty, 2e, or 3e; got '${SOURCE_LABEL}'" >&2
      exit 2
      ;;
  esac
  RUN_NAME="${RUN_NAME:-${MODEL}_${SOURCE_LABEL}_${EVENTS_PER_SOURCE}events_${SLURM_JOB_ID:-manual}}"
else
  RUN_NAME="${RUN_NAME:-${MODEL}_${EVENTS_PER_SOURCE}per_source_${SLURM_JOB_ID:-manual}}"
fi

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}"
mkdir -p outputs/slurm "${OUTPUT_ROOT}"
source "${VENV_DIR:-${REPO_ROOT}/.venv}/bin/activate"
python -m pip install -e .

echo "host: $(hostname)"
echo "repo: ${REPO_ROOT}"
echo "model: ${MODEL}"
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
echo "events per source: ${EVENTS_PER_SOURCE}"
echo "ECal energy transform: ${ECAL_ENERGY_TRANSFORM}"
echo "TPAD PE transform: ${TPAD_PE_TRANSFORM}"
if [[ -n "${SOURCE_LABEL}" ]]; then
  echo "single source: ${SOURCE_LABEL} (${ELECTRON_COUNT} electrons)"
fi
python --version
nvidia-smi
python -u scripts/check_cuda_environment.py

cmd=(
  python -u scripts/train_hit_classifier_baseline.py
  --model "${MODEL}"
  --events-per-source "${EVENTS_PER_SOURCE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --dim-feedforward "${DIM_FEEDFORWARD}"
  --dropout "${DROPOUT}"
  --seed "${SEED}"
  --ecal-energy-transform "${ECAL_ENERGY_TRANSFORM}"
  --tpad-pe-transform "${TPAD_PE_TRANSFORM}"
  --device cuda
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --no-progress
)
if [[ -n "${SOURCE_LABEL}" ]]; then
  cmd+=(--processed-source "${ELECTRON_COUNT}" "${SOURCE_LABEL}" "${PROCESSED_CACHE_ROOT}/${SOURCE_LABEL}/events")
else
  cmd+=(--processed-cache-root "${PROCESSED_CACHE_ROOT}")
fi
if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

printf 'command:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
