#!/bin/bash
# Submit one-epoch 2e Transformer jobs to find a practical A100 batch size.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-${REPO_ROOT}/data/processed/production_10M_001_sharded}"
CAMPAIGN="${CAMPAIGN:-transformer_2e_batch_sweep_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cosmos_batch_sweeps/${CAMPAIGN}}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/sbatch/cosmos_train_baseline.sbatch"
BATCH_SIZES=(2048 1024 512 256 128 64 32)
EVENTS_PER_SOURCE=100000
EPOCHS=1
EARLY_STOPPING_MIN_EPOCHS=1
EARLY_STOPPING_PATIENCE=0
EARLY_STOPPING_MIN_DELTA=1e-4
LR=3e-4
WEIGHT_DECAY=1e-4
HIDDEN_DIM=128
NUM_LAYERS=3
NUM_HEADS=4
DIM_FEEDFORWARD=256
DROPOUT=0.1
GRAD_CLIP=1.0
SHARD_CACHE_SIZE=1
CACHE_MODEL_VIEWS=1
NUM_ECAL_PLOTS=0
NUM_DIAGNOSTIC_EVENT_DISPLAYS=0
GPU_MONITOR_INTERVAL=5
SEED=7
ECAL_ENERGY_TRANSFORM=log1p
TPAD_PE_TRANSFORM=log1p

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${REPO_ROOT}/.git" ]] || fail "Not a Git checkout: ${REPO_ROOT}"
[[ -f "${TRAIN_SCRIPT}" ]] || fail "Missing training wrapper: ${TRAIN_SCRIPT}"
[[ "${CAMPAIGN}" != */* ]] || fail "CAMPAIGN must be one directory name, not a path: ${CAMPAIGN}"

cd "${REPO_ROOT}"
git diff --check
if [[ -n "$(git status --short)" && "${ALLOW_DIRTY_REPO}" != "1" ]]; then
  git status --short >&2
  fail "Refusing to submit from a dirty checkout. Commit/sync the intended training code first."
fi

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_BRANCH="$(git branch --show-current)"

if [[ "${DRY_RUN}" != "1" ]]; then
  command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable; run this launcher on COSMOS."
  [[ -x "${VENV_DIR}/bin/python" ]] || fail "Missing prepared Python environment: ${VENV_DIR}"

  if command -v module >/dev/null 2>&1; then
    module --force purge
    module load SoftwareTree/Milan
    module load GCC/13.2.0
    module load Python/3.11.5
  fi

  "${VENV_DIR}/bin/python" - "${PROCESSED_CACHE_ROOT}" "${EVENTS_PER_SOURCE}" <<'PY'
import sys
from pathlib import Path

from ml_ldmx.datasets.ecal_tpad_shards import validate_ml_ready_sharded_cache

cache = Path(sys.argv[1]).resolve() / "2e/events"
required = int(sys.argv[2])
manifest, index = validate_ml_ready_sharded_cache(cache)
spec = manifest["cache_spec"]
count = int(index["num_events"])
if count < required:
    raise RuntimeError(f"2e cache has {count:,} events; need at least {required:,}.")
if spec.get("ecal_energy_transform") != "log1p":
    raise RuntimeError("2e cache does not store log1p ECal energy.")
if spec.get("tpad_pe_transform") != "log1p":
    raise RuntimeError("2e cache does not store log1p TPad pe.")
print(f"validated 2e: {count:,} events in {len(index['shards']):,} shards")
PY
fi

[[ ! -e "${OUTPUT_ROOT}" ]] || fail "Campaign output already exists: ${OUTPUT_ROOT}"
mkdir -p "${REPO_ROOT}/outputs/slurm" "${OUTPUT_ROOT}"

MANIFEST_FILE="${OUTPUT_ROOT}/campaign_manifest.txt"
JOBS_FILE="${OUTPUT_ROOT}/submitted_jobs.tsv"
printf 'job_id\tjob_name\trun_name\tbatch_size\tevents\tepochs\n' > "${JOBS_FILE}"
{
  printf 'campaign=%s\n' "${CAMPAIGN}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'git_branch=%s\n' "${GIT_BRANCH}"
  printf 'processed_cache_root=%s\n' "${PROCESSED_CACHE_ROOT}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'model=ECalTpadTransformer\n'
  printf 'source_label=2e\n'
  printf 'events=%s\n' "${EVENTS_PER_SOURCE}"
  printf 'epochs=%s\n' "${EPOCHS}"
  printf 'batch_sizes_descending=%s\n' "${BATCH_SIZES[*]}"
  printf 'learning_rate=%s\n' "${LR}"
  printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
  printf 'hidden_dim=%s\n' "${HIDDEN_DIM}"
  printf 'num_layers=%s\n' "${NUM_LAYERS}"
  printf 'num_heads=%s\n' "${NUM_HEADS}"
  printf 'dim_feedforward=%s\n' "${DIM_FEEDFORWARD}"
  printf 'seed=%s\n' "${SEED}"
} > "${MANIFEST_FILE}"

unset RESUME
for batch_size in "${BATCH_SIZES[@]}"; do
  job_name="ml_t2_bs${batch_size}"
  run_name="transformer_2e_100k_h128_l3_b${batch_size}_seed7_benchmark"
  export_spec="ALL,REPO_ROOT=${REPO_ROOT},VENV_DIR=${VENV_DIR},PROCESSED_CACHE_ROOT=${PROCESSED_CACHE_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL=ECalTpadTransformer,SOURCE_LABEL=2e,ELECTRON_COUNT=2,EVENTS_PER_SOURCE=${EVENTS_PER_SOURCE},EPOCHS=${EPOCHS},EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA},BATCH_SIZE=${batch_size},LR=${LR},WEIGHT_DECAY=${WEIGHT_DECAY},HIDDEN_DIM=${HIDDEN_DIM},NUM_LAYERS=${NUM_LAYERS},NUM_HEADS=${NUM_HEADS},DIM_FEEDFORWARD=${DIM_FEEDFORWARD},DROPOUT=${DROPOUT},GRAD_CLIP=${GRAD_CLIP},SHARD_CACHE_SIZE=${SHARD_CACHE_SIZE},CACHE_MODEL_VIEWS=${CACHE_MODEL_VIEWS},NUM_ECAL_PLOTS=${NUM_ECAL_PLOTS},NUM_DIAGNOSTIC_EVENT_DISPLAYS=${NUM_DIAGNOSTIC_EVENT_DISPLAYS},GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL},SEED=${SEED},ECAL_ENERGY_TRANSFORM=${ECAL_ENERGY_TRANSFORM},TPAD_PE_TRANSFORM=${TPAD_PE_TRANSFORM},RUN_NAME=${run_name}"
  sbatch_args=(
    --parsable
    --chdir="${REPO_ROOT}"
    --job-name="${job_name}"
    --time=08:00:00
    --mem=64G
    --export="${export_spec}"
  )
  if [[ -n "${SBATCH_ACCOUNT}" ]]; then
    sbatch_args+=(--account="${SBATCH_ACCOUNT}")
  fi
  sbatch_args+=("${TRAIN_SCRIPT}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' sbatch "${sbatch_args[@]}"
    printf '\n'
    job_id=DRY_RUN
  else
    submission="$(sbatch "${sbatch_args[@]}")"
    job_id="${submission%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response for batch ${batch_size}: ${submission}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${job_name}" "${run_name}" "${batch_size}" \
    "${EVENTS_PER_SOURCE}" "${EPOCHS}" | tee -a "${JOBS_FILE}"
done

printf 'campaign=%s\n' "${CAMPAIGN}"
printf 'output_root=%s\n' "${OUTPUT_ROOT}"
printf 'jobs_file=%s\n' "${JOBS_FILE}"
printf 'Jobs were submitted in descending batch-size order; Slurm may start them in a different order.\n'
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'No jobs were submitted because DRY_RUN=1.\n'
fi
