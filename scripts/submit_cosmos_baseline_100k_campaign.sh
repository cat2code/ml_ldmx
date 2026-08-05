#!/bin/bash
# Submit the four report-ready 100k COSMOS baseline training jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-${REPO_ROOT}/data/processed/production_10M_001_sharded}"
CAMPAIGN="${CAMPAIGN:-report_baselines_100k_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cosmos_baselines/${CAMPAIGN}}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/sbatch/cosmos_train_baseline.sbatch"
EVENTS_PER_SOURCE=100000
EPOCHS=15
EARLY_STOPPING_MIN_EPOCHS=5
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_MIN_DELTA=1e-4
BATCH_SIZE=8
LR=3e-4
WEIGHT_DECAY=1e-4
DROPOUT=0.1
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
printf 'job_id\tjob_name\trun_name\tmodel\tsource_label\tevents\n' > "${JOBS_FILE}"
{
  printf 'campaign=%s\n' "${CAMPAIGN}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'git_branch=%s\n' "${GIT_BRANCH}"
  printf 'processed_cache_root=%s\n' "${PROCESSED_CACHE_ROOT}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'events_per_run=%s\n' "${EVENTS_PER_SOURCE}"
  printf 'epochs_max=%s\n' "${EPOCHS}"
  printf 'early_stopping_min_epochs=%s\n' "${EARLY_STOPPING_MIN_EPOCHS}"
  printf 'early_stopping_patience=%s\n' "${EARLY_STOPPING_PATIENCE}"
  printf 'early_stopping_min_delta=%s\n' "${EARLY_STOPPING_MIN_DELTA}"
  printf 'batch_size=%s\n' "${BATCH_SIZE}"
  printf 'learning_rate=%s\n' "${LR}"
  printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
  printf 'seed=%s\n' "${SEED}"
} > "${MANIFEST_FILE}"

unset RESUME

submit_run() {
  local job_name=$1
  local model=$2
  local source_label=$3
  local electron_count=$4
  local hidden_dim=$5
  local num_layers=$6
  local num_heads=$7
  local dim_feedforward=$8
  local space_dimensions=$9
  local propagate_dimensions=${10}
  local k=${11}
  local run_name=${12}
  local export_spec submission job_id
  local -a sbatch_args

  export_spec="ALL,REPO_ROOT=${REPO_ROOT},VENV_DIR=${VENV_DIR},PROCESSED_CACHE_ROOT=${PROCESSED_CACHE_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL=${model},SOURCE_LABEL=${source_label},ELECTRON_COUNT=${electron_count},EVENTS_PER_SOURCE=${EVENTS_PER_SOURCE},EPOCHS=${EPOCHS},EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA},BATCH_SIZE=${BATCH_SIZE},LR=${LR},WEIGHT_DECAY=${WEIGHT_DECAY},HIDDEN_DIM=${hidden_dim},NUM_LAYERS=${num_layers},NUM_HEADS=${num_heads},DIM_FEEDFORWARD=${dim_feedforward},DROPOUT=${DROPOUT},SPACE_DIMENSIONS=${space_dimensions},PROPAGATE_DIMENSIONS=${propagate_dimensions},K=${k},GRAD_CLIP=${GRAD_CLIP},SHARD_CACHE_SIZE=${SHARD_CACHE_SIZE},CACHE_MODEL_VIEWS=${CACHE_MODEL_VIEWS},SEED=${SEED},ECAL_ENERGY_TRANSFORM=${ECAL_ENERGY_TRANSFORM},TPAD_PE_TRANSFORM=${TPAD_PE_TRANSFORM},RUN_NAME=${run_name}"
  sbatch_args=(
    --parsable
    --chdir="${REPO_ROOT}"
    --job-name="${job_name}"
    --time=72:00:00
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
    job_id="DRY_RUN"
  else
    submission="$(sbatch "${sbatch_args[@]}")"
    job_id="${submission%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response for ${job_name}: ${submission}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${job_name}" "${run_name}" "${model}" "${source_label}" "${EVENTS_PER_SOURCE}" \
    | tee -a "${JOBS_FILE}"
}

submit_run ml_t2_100k ECalTpadTransformer 2e 2 128 3 4 256 4 128 16 \
  transformer_2e_100k_h128_l3_seed7
submit_run ml_t3_100k ECalTpadTransformer 3e 3 128 3 4 256 4 128 16 \
  transformer_3e_100k_h128_l3_seed7
submit_run ml_g2_100k ECalTpadGravNet 2e 2 128 4 4 256 4 128 16 \
  gravnet_2e_100k_h128_l4_k16_seed7
submit_run ml_g3_100k ECalTpadGravNet 3e 3 128 4 4 256 4 128 16 \
  gravnet_3e_100k_h128_l4_k16_seed7

printf 'campaign=%s\n' "${CAMPAIGN}"
printf 'output_root=%s\n' "${OUTPUT_ROOT}"
printf 'jobs_file=%s\n' "${JOBS_FILE}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'No jobs were submitted because DRY_RUN=1.\n'
fi
