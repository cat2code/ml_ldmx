#!/bin/bash
#SBATCH -J ml_ldmx_baseline
#SBATCH -p gpua100i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH --no-requeue
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-data/processed/production_10M_001_sharded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cosmos_baselines}"
MODEL="${MODEL:-ECalTpadTransformer}"
SOURCE_LABEL="${SOURCE_LABEL:-}"              # leave empty for balanced 2e+3e; set to 2e or 3e for one source
ELECTRON_COUNT="${ELECTRON_COUNT:-}"          # inferred from SOURCE_LABEL when omitted
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-500}" # total events = sources * this value
EPOCHS="${EPOCHS:-15}"
EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-128}"
DROPOUT="${DROPOUT:-0.1}"
SPACE_DIMENSIONS="${SPACE_DIMENSIONS:-4}"
PROPAGATE_DIMENSIONS="${PROPAGATE_DIMENSIONS:-32}"
K="${K:-16}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-1}"
CACHE_MODEL_VIEWS="${CACHE_MODEL_VIEWS:-0}"
NUM_ECAL_PLOTS="${NUM_ECAL_PLOTS:-2}"
NUM_DIAGNOSTIC_EVENT_DISPLAYS="${NUM_DIAGNOSTIC_EVENT_DISPLAYS:-3}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"
SEED="${SEED:-7}"
RESUME="${RESUME:-}"
ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM:-log1p}"
TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM:-log1p}"
DEVICE="${DEVICE:-cuda}"

if [[ -n "${SOURCE_LABEL}" ]]; then
  case "${SOURCE_LABEL}" in
    2e)
      [[ -z "${ELECTRON_COUNT}" || "${ELECTRON_COUNT}" == "2" ]] || {
        echo "SOURCE_LABEL=2e requires ELECTRON_COUNT=2; got '${ELECTRON_COUNT}'" >&2
        exit 2
      }
      ELECTRON_COUNT=2
      VALID_LABELS=(1 2)
      ;;
    3e)
      [[ -z "${ELECTRON_COUNT}" || "${ELECTRON_COUNT}" == "3" ]] || {
        echo "SOURCE_LABEL=3e requires ELECTRON_COUNT=3; got '${ELECTRON_COUNT}'" >&2
        exit 2
      }
      ELECTRON_COUNT=3
      VALID_LABELS=(1 2 3)
      ;;
    *)
      echo "SOURCE_LABEL must be empty, 2e, or 3e; got '${SOURCE_LABEL}'" >&2
      exit 2
      ;;
  esac
  RUN_NAME="${RUN_NAME:-${MODEL}_${SOURCE_LABEL}_${EVENTS_PER_SOURCE}events_${SLURM_JOB_ID:-manual}}"
else
  VALID_LABELS=(1 2 3)
  RUN_NAME="${RUN_NAME:-${MODEL}_${EVENTS_PER_SOURCE}per_source_${SLURM_JOB_ID:-manual}}"
fi

if [[ "${DEVICE}" != "cuda" && "${DEVICE}" != "cpu" ]]; then
  echo "DEVICE must be cuda or cpu; got '${DEVICE}'" >&2
  exit 2
fi
if [[ "${DEVICE}" == "cuda" && ! "${GPU_MONITOR_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_MONITOR_INTERVAL must be a positive integer; got '${GPU_MONITOR_INTERVAL}'" >&2
  exit 2
fi

module --force purge
module load SoftwareTree/Milan
module load GCC/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}"
mkdir -p outputs/slurm "${OUTPUT_ROOT}"
source "${VENV_DIR:-${REPO_ROOT}/.venv}/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

JOB_STARTED_AT="$(date --iso-8601=seconds)"
JOB_STARTED_EPOCH="$(date +%s)"
GPU_MONITOR_PID=""
GPU_LOG="outputs/slurm/${SLURM_JOB_NAME:-ml_ldmx_baseline}_${SLURM_JOB_ID:-manual}_gpu.csv"

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
echo "model: ${MODEL}"
echo "device: ${DEVICE}"
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
echo "events per source: ${EVENTS_PER_SOURCE}"
echo "valid labels: ${VALID_LABELS[*]}"
echo "ECal energy transform: ${ECAL_ENERGY_TRANSFORM}"
echo "TPAD PE transform: ${TPAD_PE_TRANSFORM}"
if [[ -n "${SOURCE_LABEL}" ]]; then
  echo "single source: ${SOURCE_LABEL} (${ELECTRON_COUNT} electrons)"
fi
python --version
if [[ "${DEVICE}" == "cuda" ]]; then
  nvidia-smi
  python -u tests/check_cuda_environment.py
fi
python -c "import ml_ldmx, numpy, torch, torch_geometric; print('Training dependencies are available')"
if [[ "${MODEL}" == *GravNet ]]; then
  python -c "import torch_cluster, torch_scatter; print('torch-cluster and torch-scatter are available')"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  scontrol show job "${SLURM_JOB_ID}" || true
fi
if [[ "${DEVICE}" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_monitor_cmd=(nvidia-smi)
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_monitor_cmd+=(--id="${CUDA_VISIBLE_DEVICES}")
  fi
  gpu_monitor_cmd+=(
    "--query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    --format=csv
    -l "${GPU_MONITOR_INTERVAL}"
  )
  "${gpu_monitor_cmd[@]}" > "${GPU_LOG}" 2>&1 &
  GPU_MONITOR_PID=$!
  echo "GPU telemetry: ${GPU_LOG}"
fi

cmd=(
  python -u scripts/train_hit_classifier_baseline.py
  --model "${MODEL}"
  --events-per-source "${EVENTS_PER_SOURCE}"
  --epochs "${EPOCHS}"
  --early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --dim-feedforward "${DIM_FEEDFORWARD}"
  --dropout "${DROPOUT}"
  --space-dimensions "${SPACE_DIMENSIONS}"
  --propagate-dimensions "${PROPAGATE_DIMENSIONS}"
  --k "${K}"
  --grad-clip "${GRAD_CLIP}"
  --shard-cache-size "${SHARD_CACHE_SIZE}"
  --seed "${SEED}"
  --valid-labels "${VALID_LABELS[@]}"
  --ecal-energy-transform "${ECAL_ENERGY_TRANSFORM}"
  --tpad-pe-transform "${TPAD_PE_TRANSFORM}"
  --device "${DEVICE}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --num-ecal-plots "${NUM_ECAL_PLOTS}"
  --num-diagnostic-event-displays "${NUM_DIAGNOSTIC_EVENT_DISPLAYS}"
  --no-progress
)
if [[ "${CACHE_MODEL_VIEWS}" == "1" ]]; then
  cmd+=(--cache-model-views)
fi
if [[ -n "${SOURCE_LABEL}" ]]; then
  cmd+=(--processed-source "${ELECTRON_COUNT}" "${SOURCE_LABEL}" "${PROCESSED_CACHE_ROOT}/${SOURCE_LABEL}/events")
else
  cmd+=(--processed-cache-root "${PROCESSED_CACHE_ROOT}")
fi
if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

printf 'command:'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -v "${cmd[@]}"
else
  "${cmd[@]}"
fi
