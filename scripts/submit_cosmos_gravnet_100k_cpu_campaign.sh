#!/bin/bash
# Submit CPU duplicates of the normalized 2e and 3e GravNet 100k runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-${REPO_ROOT}/data/processed/production_10M_001_sharded}"
CAMPAIGN="${CAMPAIGN:-report_normalized_gravnet_cpu_100k_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cosmos_baselines/${CAMPAIGN}}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/sbatch/cosmos_train_baseline_cpu.sbatch"
EVENTS_PER_SOURCE=100000
EPOCHS=15
EARLY_STOPPING_MIN_EPOCHS=5
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_MIN_DELTA=1e-4
BATCH_SIZE=32
LR=3e-4
WEIGHT_DECAY=1e-4
HIDDEN_DIM=128
NUM_LAYERS=4
NUM_HEADS=4
DIM_FEEDFORWARD=256
DROPOUT=0.1
SPACE_DIMENSIONS=4
PROPAGATE_DIMENSIONS=128
K=16
GRAD_CLIP=1.0
SHARD_CACHE_SIZE=1
CACHE_MODEL_VIEWS=1
SEED=7
ECAL_ENERGY_TRANSFORM=log1p
TPAD_PE_TRANSFORM=log1p

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${REPO_ROOT}/.git" ]] || fail "Not a Git checkout: ${REPO_ROOT}"
[[ -f "${TRAIN_SCRIPT}" ]] || fail "Missing CPU training wrapper: ${TRAIN_SCRIPT}"
[[ "${CAMPAIGN}" != */* ]] || fail "CAMPAIGN must be one directory name, not a path."

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

  module --force purge
  module load SoftwareTree/Milan
  module load GCC/13.2.0
  module load Python/3.11.5

  "${VENV_DIR}/bin/python" - "${PROCESSED_CACHE_ROOT}" <<'PY'
import sys
from pathlib import Path

from ml_ldmx.datasets.ecal_tpad_shards import validate_ml_ready_sharded_cache

root = Path(sys.argv[1]).resolve()
for label in ("2e", "3e"):
    manifest, index = validate_ml_ready_sharded_cache(root / label / "events")
    spec = manifest["cache_spec"]
    count = int(index["num_events"])
    if count < 100_000:
        raise RuntimeError(f"{label} cache has {count:,} events; need at least 100,000.")
    if spec.get("ecal_energy_transform") != "log1p":
        raise RuntimeError(f"{label} cache does not store log1p ECal energy.")
    if spec.get("tpad_pe_transform") != "log1p":
        raise RuntimeError(f"{label} cache does not store log1p TPad pe.")
    print(f"validated {label}: {count:,} events in {len(index['shards']):,} shards")
PY
fi

[[ ! -e "${OUTPUT_ROOT}" ]] || fail "Campaign output already exists: ${OUTPUT_ROOT}"
mkdir -p "${REPO_ROOT}/outputs/slurm" "${OUTPUT_ROOT}"

MANIFEST_FILE="${OUTPUT_ROOT}/campaign_manifest.txt"
JOBS_FILE="${OUTPUT_ROOT}/submitted_jobs.tsv"
printf 'job_id\tresource\tjob_name\trun_name\tmodel\tsource_label\tevents\tbatch_size\n' > "${JOBS_FILE}"
{
  printf 'campaign=%s\n' "${CAMPAIGN}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'git_branch=%s\n' "${GIT_BRANCH}"
  printf 'processed_cache_root=%s\n' "${PROCESSED_CACHE_ROOT}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'resource=cpu_lu48_16_threads_64G\n'
  printf 'events_per_run=%s\n' "${EVENTS_PER_SOURCE}"
  printf 'epochs_max=%s\n' "${EPOCHS}"
  printf 'early_stopping_min_epochs=%s\n' "${EARLY_STOPPING_MIN_EPOCHS}"
  printf 'early_stopping_patience=%s\n' "${EARLY_STOPPING_PATIENCE}"
  printf 'early_stopping_min_delta=%s\n' "${EARLY_STOPPING_MIN_DELTA}"
  printf 'architecture=ECalTpadGravNet_h%s_l%s_space%s_propagate%s_k%s\n' \
    "${HIDDEN_DIM}" "${NUM_LAYERS}" "${SPACE_DIMENSIONS}" \
    "${PROPAGATE_DIMENSIONS}" "${K}"
  printf 'gravnet_normalization=batchnorm_after_each_residual_block\n'
  printf 'batch_size=%s\n' "${BATCH_SIZE}"
  printf 'learning_rate=%s\n' "${LR}"
  printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
  printf 'dropout=%s\n' "${DROPOUT}"
  printf 'seed=%s\n' "${SEED}"
  printf 'dependency_policy=none\n'
  printf 'gpu_jobs_cancelled=false\n'
} > "${MANIFEST_FILE}"

submit_run() {
  local source_label=$1
  local electron_count=$2
  local job_name="ml_g${electron_count}n_c100k"
  local run_name="gravnet_${source_label}_100k_h128_l4_k16_batchnorm_cpu_b32_seed7"
  local export_spec submission job_id
  local -a sbatch_args

  export_spec="ALL,REPO_ROOT=${REPO_ROOT},VENV_DIR=${VENV_DIR},PROCESSED_CACHE_ROOT=${PROCESSED_CACHE_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL=ECalTpadGravNet,SOURCE_LABEL=${source_label},ELECTRON_COUNT=${electron_count},EVENTS_PER_SOURCE=${EVENTS_PER_SOURCE},EPOCHS=${EPOCHS},EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA},BATCH_SIZE=${BATCH_SIZE},LR=${LR},WEIGHT_DECAY=${WEIGHT_DECAY},HIDDEN_DIM=${HIDDEN_DIM},NUM_LAYERS=${NUM_LAYERS},NUM_HEADS=${NUM_HEADS},DIM_FEEDFORWARD=${DIM_FEEDFORWARD},DROPOUT=${DROPOUT},SPACE_DIMENSIONS=${SPACE_DIMENSIONS},PROPAGATE_DIMENSIONS=${PROPAGATE_DIMENSIONS},K=${K},GRAD_CLIP=${GRAD_CLIP},SHARD_CACHE_SIZE=${SHARD_CACHE_SIZE},CACHE_MODEL_VIEWS=${CACHE_MODEL_VIEWS},SEED=${SEED},ECAL_ENERGY_TRANSFORM=${ECAL_ENERGY_TRANSFORM},TPAD_PE_TRANSFORM=${TPAD_PE_TRANSFORM},RUN_NAME=${run_name}"
  sbatch_args=(
    --parsable
    --chdir="${REPO_ROOT}"
    --job-name="${job_name}"
    --time=72:00:00
    --cpus-per-task=16
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
    job_id="DRY_RUN_${electron_count}e"
  else
    submission="$(sbatch "${sbatch_args[@]}")"
    job_id="${submission%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response for ${job_name}: ${submission}"
  fi

  printf '%s\tcpu\t%s\t%s\tECalTpadGravNet\t%s\t%s\t%s\n' \
    "${job_id}" "${job_name}" "${run_name}" "${source_label}" \
    "${EVENTS_PER_SOURCE}" "${BATCH_SIZE}" | tee -a "${JOBS_FILE}"
}

submit_run 2e 2
submit_run 3e 3

printf 'campaign=%s\n' "${CAMPAIGN}"
printf 'output_root=%s\n' "${OUTPUT_ROOT}"
printf 'jobs_file=%s\n' "${JOBS_FILE}"
printf 'Submitted independent CPU duplicates; existing GPU jobs were not modified.\n'
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'No jobs were submitted because DRY_RUN=1.\n'
fi
