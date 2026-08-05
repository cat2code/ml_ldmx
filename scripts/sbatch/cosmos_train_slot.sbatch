#!/bin/bash
#SBATCH -J ml_ldmx_slot
#SBATCH -p gpua100i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH --no-requeue
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-data/processed/production_10M_001_sharded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cosmos_slot}"
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-500}"  # total events = 2 * this for 2e+3e
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
LAMBDA_ORIGIN="${LAMBDA_ORIGIN:-1.0}"
LAMBDA_FRACTION="${LAMBDA_FRACTION:-1.0}"
LAMBDA_SLOT="${LAMBDA_SLOT:-0.5}"
LAMBDA_COUNT="${LAMBDA_COUNT:-1.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-plateau}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-1}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-1}"
SEED="${SEED:-7}"
SUPERVISE_NOISE="${SUPERVISE_NOISE:-1}"
RUN_NAME="${RUN_NAME:-slot_${EVENTS_PER_SOURCE}per_source_${SLURM_JOB_ID:-manual}}"
RESUME="${RESUME:-}"
ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM:-log1p}"
TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM:-log1p}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_EVENTS_PER_SOURCE="${PREFLIGHT_EVENTS_PER_SOURCE:-50}"

module --force purge
module load SoftwareTree/Milan
module load GCC/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}"
mkdir -p outputs/slurm "${OUTPUT_ROOT}"
source "${VENV_DIR:-${REPO_ROOT}/.venv}/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

JOB_STARTED_AT="$(date --iso-8601=seconds)"
JOB_STARTED_EPOCH="$(date +%s)"
GPU_MONITOR_PID=""
GPU_LOG="outputs/slurm/${SLURM_JOB_NAME:-ml_ldmx_slot}_${SLURM_JOB_ID:-manual}_gpu.csv"

finish_job() {
  status=$?
  trap - EXIT
  set +e
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null
    wait "${GPU_MONITOR_PID}" 2>/dev/null
  fi
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
echo "started: ${JOB_STARTED_AT}"
echo "Slurm job: ${SLURM_JOB_ID:-manual} (${SLURM_JOB_NAME:-manual})"
echo "repo: ${REPO_ROOT}"
echo "git commit: $(git rev-parse HEAD)"
echo "git branch: $(git branch --show-current)"
if [[ -n "$(git status --short)" ]]; then
  echo "WARNING: repository worktree is not clean:"
  git status --short
fi
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
echo "events per source: ${EVENTS_PER_SOURCE} (total=$((2 * EVENTS_PER_SOURCE)))"
echo "architecture: hidden=${HIDDEN_DIM} layers=${NUM_LAYERS} heads=${NUM_HEADS} dropout=${DROPOUT}"
echo "noise supervision: ${SUPERVISE_NOISE}"
echo "ECal energy transform: ${ECAL_ENERGY_TRANSFORM}"
echo "TPad PE transform: ${TPAD_PE_TRANSFORM}"
python --version
nvidia-smi
python -u tests/check_cuda_environment.py
python -c "import matplotlib, ml_ldmx, numpy, torch; print('Slot training dependencies are available')"
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  scontrol show job "${SLURM_JOB_ID}" || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_monitor_cmd=(nvidia-smi)
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_monitor_cmd+=(--id="${CUDA_VISIBLE_DEVICES}")
  fi
  gpu_monitor_cmd+=(
    "--query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    --format=csv
    -l 60
  )
  "${gpu_monitor_cmd[@]}" > "${GPU_LOG}" 2>&1 &
  GPU_MONITOR_PID=$!
  echo "GPU telemetry: ${GPU_LOG}"
fi

common_args=(
  --processed-cache-root "${PROCESSED_CACHE_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --dropout "${DROPOUT}"
  --lambda-origin "${LAMBDA_ORIGIN}"
  --lambda-fraction "${LAMBDA_FRACTION}"
  --lambda-slot "${LAMBDA_SLOT}"
  --lambda-count "${LAMBDA_COUNT}"
  --grad-clip "${GRAD_CLIP}"
  --lr-scheduler "${LR_SCHEDULER}"
  --plateau-patience "${PLATEAU_PATIENCE}"
  --plateau-factor "${PLATEAU_FACTOR}"
  --shard-cache-size "${SHARD_CACHE_SIZE}"
  --seed "${SEED}"
  --ecal-energy-transform "${ECAL_ENERGY_TRANSFORM}"
  --tpad-pe-transform "${TPAD_PE_TRANSFORM}"
  --device cuda
  --no-progress
)
if [[ "${SUPERVISE_NOISE}" == "1" ]]; then
  common_args+=(--supervise-noise)
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  preflight_name="${RUN_NAME}_preflight_${SLURM_JOB_ID:-manual}"
  preflight_cmd=(
    python -u scripts/train_ecal_tpad_slot_model.py
    "${common_args[@]}"
    --events-per-source "${PREFLIGHT_EVENTS_PER_SOURCE}"
    --epochs 1
    --early-stopping-min-epochs 1
    --early-stopping-patience 0
    --output-root "${OUTPUT_ROOT}/_preflight"
    --run-name "${preflight_name}"
    --num-ecal-plots 0
  )
  printf 'preflight command:'; printf ' %q' "${preflight_cmd[@]}"; printf '\n'
  "${preflight_cmd[@]}"
  echo "Slot-model end-to-end preflight passed; starting the production run."
fi

cmd=(
  python -u scripts/train_ecal_tpad_slot_model.py
  "${common_args[@]}"
  --events-per-source "${EVENTS_PER_SOURCE}"
  --epochs "${EPOCHS}"
  --early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
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
