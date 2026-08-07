#!/bin/bash
#SBATCH -J ml_contributor_slot_cpu
#SBATCH -p lu48
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH --no-requeue
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

# CPU-only COSMOS wrapper for ECalTpadContributorSetSlotModel.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-data/processed/production_10M_001_sharded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cosmos_contributor_set_slot}"
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-5000}"
EPOCHS="${EPOCHS:-15}"
EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"
LAMBDA_SUPPORT="${LAMBDA_SUPPORT:-1.0}"
LAMBDA_FRACTION="${LAMBDA_FRACTION:-1.0}"
LAMBDA_SLOT="${LAMBDA_SLOT:-1.0}"
CONTRIBUTION_EPSILON="${CONTRIBUTION_EPSILON:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-1}"
SEED="${SEED:-7}"
RUN_NAME="${RUN_NAME:-contributor_slot_cpu_${SLURM_JOB_ID:-manual}}"
RESUME="${RESUME:-}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_EVENTS_PER_SOURCE="${PREFLIGHT_EVENTS_PER_SOURCE:-50}"
MAX_PLOT_HITS="${MAX_PLOT_HITS:-100000}"

module --force purge
module load SoftwareTree/Milan
module load GCC/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}"
mkdir -p outputs/slurm "${OUTPUT_ROOT}"
source "${VENV_DIR}/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

JOB_STARTED_EPOCH="$(date +%s)"
finish_job() {
  status=$?
  trap - EXIT
  set +e
  finished_epoch="$(date +%s)"
  echo "finished: $(date --iso-8601=seconds)"
  echo "wall seconds: $((finished_epoch - JOB_STARTED_EPOCH))"
  echo "exit status: ${status}"
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sstat >/dev/null 2>&1; then
    sstat -j "${SLURM_JOB_ID}.batch" \
      --format=JobID,AveCPU,MaxRSS,MaxVMSize,AveDiskRead,AveDiskWrite -P || true
  fi
  exit "${status}"
}
trap finish_job EXIT

echo "host: $(hostname)"
echo "started: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID:-manual} (${SLURM_JOB_NAME:-manual})"
echo "repo: ${REPO_ROOT}"
echo "git commit: $(git rev-parse HEAD)"
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
echo "events: ${EVENTS_PER_SOURCE} per source ($((2 * EVENTS_PER_SOURCE)) total)"
echo "architecture: hidden=${HIDDEN_DIM} layers=${NUM_LAYERS} heads=${NUM_HEADS}"
echo "batch size: ${BATCH_SIZE}"
echo "CPU threads: ${OMP_NUM_THREADS}"
python --version
python -c "import matplotlib, ml_ldmx, numpy, torch; from ml_ldmx.models import ECalTpadContributorSetSlotModel; print('CPU contributor-set dependencies are available; torch threads:', torch.get_num_threads())"

common_args=(
  --processed-cache-root "${PROCESSED_CACHE_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --dropout "${DROPOUT}"
  --lambda-support "${LAMBDA_SUPPORT}"
  --lambda-fraction "${LAMBDA_FRACTION}"
  --lambda-slot "${LAMBDA_SLOT}"
  --contribution-epsilon "${CONTRIBUTION_EPSILON}"
  --grad-clip "${GRAD_CLIP}"
  --shard-cache-size "${SHARD_CACHE_SIZE}"
  --seed "${SEED}"
  --ecal-energy-transform log1p
  --tpad-pe-transform log1p
  --device cpu
  --no-progress
)

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  preflight_cmd=(
    python -u scripts/train_ecal_tpad_contributor_set_slot_model.py
    "${common_args[@]}"
    --events-per-source "${PREFLIGHT_EVENTS_PER_SOURCE}"
    --epochs 1
    --early-stopping-min-epochs 1
    --early-stopping-patience 0
    --output-root "${OUTPUT_ROOT}/_preflight"
    --run-name "${RUN_NAME}_preflight"
    --num-ecal-plots 0
    --max-plot-hits 1000
  )
  printf 'preflight command:'; printf ' %q' "${preflight_cmd[@]}"; printf '\n'
  "${preflight_cmd[@]}"
  echo "CPU contributor-set preflight passed; starting production training."
fi

cmd=(
  python -u scripts/train_ecal_tpad_contributor_set_slot_model.py
  "${common_args[@]}"
  --events-per-source "${EVENTS_PER_SOURCE}"
  --epochs "${EPOCHS}"
  --early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --max-plot-hits "${MAX_PLOT_HITS}"
)
if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

printf 'command:'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -v "${cmd[@]}"
else
  "${cmd[@]}"
fi
