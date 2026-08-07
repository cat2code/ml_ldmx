#!/bin/bash
# Submit independent contributor-set slot-model scaling runs on COSMOS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-${REPO_ROOT}/data/processed/production_10M_001_sharded}"
CAMPAIGN="${CAMPAIGN:-contributor_slot_scaling_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cosmos_contributor_set_slot/${CAMPAIGN}}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

CPU_TRAIN_SCRIPT="${REPO_ROOT}/scripts/sbatch/cosmos_train_contributor_set_slot_cpu.sbatch"
GPU_TRAIN_SCRIPT="${REPO_ROOT}/scripts/sbatch/cosmos_train_contributor_set_slot.sbatch"
# Values are per source: total sizes are therefore 10k, 50k, and 100k.
EVENTS_PER_SOURCE_LIST="${EVENTS_PER_SOURCE_LIST:-5000 25000 50000}"
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

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${REPO_ROOT}/.git" ]] || fail "Not a Git checkout: ${REPO_ROOT}"
[[ -f "${CPU_TRAIN_SCRIPT}" ]] || fail "Missing CPU training wrapper: ${CPU_TRAIN_SCRIPT}"
[[ -f "${GPU_TRAIN_SCRIPT}" ]] || fail "Missing GPU training wrapper: ${GPU_TRAIN_SCRIPT}"
[[ "${CAMPAIGN}" != */* ]] || fail "CAMPAIGN must be one directory name, not a path."

read -r -a EVENTS_PER_SOURCE_VALUES <<< "${EVENTS_PER_SOURCE_LIST}"
[[ "${#EVENTS_PER_SOURCE_VALUES[@]}" -eq 3 ]] || fail \
  "EVENTS_PER_SOURCE_LIST must contain exactly three ascending integers."
previous_size=0
for size in "${EVENTS_PER_SOURCE_VALUES[@]}"; do
  [[ "${size}" =~ ^[1-9][0-9]*$ ]] || fail "Invalid per-source event count: ${size}"
  (( size > previous_size )) || fail "Dataset sizes must be strictly ascending."
  previous_size="${size}"
done
largest_per_source="${EVENTS_PER_SOURCE_VALUES[2]}"
TOTAL_EVENT_SEQUENCE=""
for size in "${EVENTS_PER_SOURCE_VALUES[@]}"; do
  TOTAL_EVENT_SEQUENCE="${TOTAL_EVENT_SEQUENCE:+${TOTAL_EVENT_SEQUENCE} }$((2 * size))"
done

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

  "${VENV_DIR}/bin/python" - "${PROCESSED_CACHE_ROOT}" "${largest_per_source}" <<'PY'
import sys
from pathlib import Path

from ml_ldmx.datasets.ecal_tpad_shards import validate_ml_ready_sharded_cache

root = Path(sys.argv[1]).resolve()
required = int(sys.argv[2])
for label in ("2e", "3e"):
    manifest, index = validate_ml_ready_sharded_cache(root / label / "events")
    spec = manifest["cache_spec"]
    count = int(index["num_events"])
    if count < required:
        raise RuntimeError(f"{label} cache has {count:,} events; need {required:,}.")
    if spec.get("ecal_energy_transform") != "log1p":
        raise RuntimeError(f"{label} cache does not store log1p ECal energy.")
    if spec.get("tpad_pe_transform") != "log1p":
        raise RuntimeError(f"{label} cache does not store log1p TPad pe.")
    if spec.get("filter_noise") is not False or spec.get("supervise_noise") is not True:
        raise RuntimeError(f"{label} cache cannot supply explicit noise targets.")
    print(f"validated {label}: {count:,} events in {len(index['shards']):,} shards")
PY
fi

[[ ! -e "${OUTPUT_ROOT}" ]] || fail "Campaign output already exists: ${OUTPUT_ROOT}"
mkdir -p "${REPO_ROOT}/outputs/slurm" "${OUTPUT_ROOT}"

MANIFEST_FILE="${OUTPUT_ROOT}/campaign_manifest.txt"
JOBS_FILE="${OUTPUT_ROOT}/submitted_jobs.tsv"
printf 'job_id\tresource\tjob_name\trun_name\ttotal_events\tevents_per_source\n' > "${JOBS_FILE}"
{
  printf 'campaign=%s\n' "${CAMPAIGN}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'git_branch=%s\n' "${GIT_BRANCH}"
  printf 'processed_cache_root=%s\n' "${PROCESSED_CACHE_ROOT}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'events_per_source_sequence=%s\n' "${EVENTS_PER_SOURCE_LIST}"
  printf 'total_event_sequence=%s\n' "${TOTAL_EVENT_SEQUENCE}"
  printf 'dependency_policy=none\n'
  printf 'submission_order=10k_cpu 50k_cpu 100k_gpu\n'
  printf 'resource_sequence=cpu cpu gpu\n'
  printf 'epochs_max=%s\n' "${EPOCHS}"
  printf 'early_stopping_min_epochs=%s\n' "${EARLY_STOPPING_MIN_EPOCHS}"
  printf 'early_stopping_patience=%s\n' "${EARLY_STOPPING_PATIENCE}"
  printf 'batch_size=%s\n' "${BATCH_SIZE}"
  printf 'architecture=ECalTpadContributorSetSlotModel_h%s_l%s_heads%s\n' \
    "${HIDDEN_DIM}" "${NUM_LAYERS}" "${NUM_HEADS}"
  printf 'learning_rate=%s\n' "${LR}"
  printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
  printf 'seed=%s\n' "${SEED}"
} > "${MANIFEST_FILE}"

run_index=0
for events_per_source in "${EVENTS_PER_SOURCE_VALUES[@]}"; do
  total_events=$((2 * events_per_source))
  if (( total_events % 1000 == 0 )); then
    size_label="$((total_events / 1000))k"
  else
    size_label="${total_events}"
  fi
  job_name="ml_cs_${size_label}"
  run_name="contributor_slot_${size_label}_h${HIDDEN_DIM}_l${NUM_LAYERS}_b${BATCH_SIZE}_seed${SEED}"
  resource="cpu"
  train_script="${CPU_TRAIN_SCRIPT}"
  if (( run_index == 2 )); then
    resource="gpu"
    train_script="${GPU_TRAIN_SCRIPT}"
  fi
  job_name="ml_cs_${resource:0:1}${size_label}"
  run_preflight=0
  if (( run_index == 0 )); then
    run_preflight=1
  fi

  export_spec="ALL,REPO_ROOT=${REPO_ROOT},VENV_DIR=${VENV_DIR},PROCESSED_CACHE_ROOT=${PROCESSED_CACHE_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},EVENTS_PER_SOURCE=${events_per_source},EPOCHS=${EPOCHS},EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA},BATCH_SIZE=${BATCH_SIZE},LR=${LR},WEIGHT_DECAY=${WEIGHT_DECAY},HIDDEN_DIM=${HIDDEN_DIM},NUM_LAYERS=${NUM_LAYERS},NUM_HEADS=${NUM_HEADS},DROPOUT=${DROPOUT},LAMBDA_SUPPORT=${LAMBDA_SUPPORT},LAMBDA_FRACTION=${LAMBDA_FRACTION},LAMBDA_SLOT=${LAMBDA_SLOT},CONTRIBUTION_EPSILON=${CONTRIBUTION_EPSILON},GRAD_CLIP=${GRAD_CLIP},SHARD_CACHE_SIZE=${SHARD_CACHE_SIZE},SEED=${SEED},RUN_PREFLIGHT=${run_preflight},RUN_NAME=${run_name}"
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
  sbatch_args+=("${train_script}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' sbatch "${sbatch_args[@]}"
    printf '\n'
    job_id="DRY_RUN_$((run_index + 1))"
  else
    submission="$(sbatch "${sbatch_args[@]}")"
    job_id="${submission%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: ${submission}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${resource}" "${job_name}" "${run_name}" \
    "${total_events}" "${events_per_source}" | tee -a "${JOBS_FILE}"
  run_index=$((run_index + 1))
done

printf 'campaign=%s\n' "${CAMPAIGN}"
printf 'output_root=%s\n' "${OUTPUT_ROOT}"
printf 'jobs_file=%s\n' "${JOBS_FILE}"
printf 'Submitted independently in this order: 10k CPU, 50k CPU, 100k GPU.\n'
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'No jobs were submitted because DRY_RUN=1.\n'
fi
