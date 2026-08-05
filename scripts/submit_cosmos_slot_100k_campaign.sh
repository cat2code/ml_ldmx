#!/bin/bash
# Submit the balanced 100k-event experimental slot-model run on COSMOS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-${REPO_ROOT}/data/processed/production_10M_001_sharded}"
CAMPAIGN="${CAMPAIGN:-report_slot_100k_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cosmos_slot/${CAMPAIGN}}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/sbatch/cosmos_train_slot.sbatch"
EVENTS_PER_SOURCE=50000
TOTAL_EVENTS=$((2 * EVENTS_PER_SOURCE))
EPOCHS=15
EARLY_STOPPING_MIN_EPOCHS=5
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_MIN_DELTA=1e-4
BATCH_SIZE=8
LR=3e-4
WEIGHT_DECAY=1e-4
HIDDEN_DIM=192
NUM_LAYERS=3
NUM_HEADS=8
DROPOUT=0.1
LAMBDA_ORIGIN=1.0
LAMBDA_FRACTION=1.0
LAMBDA_SLOT=0.5
LAMBDA_COUNT=1.0
GRAD_CLIP=1.0
LR_SCHEDULER=plateau
PLATEAU_PATIENCE=1
PLATEAU_FACTOR=0.5
SHARD_CACHE_SIZE=1
SEED=7
SUPERVISE_NOISE=1
RUN_PREFLIGHT=1
PREFLIGHT_EVENTS_PER_SOURCE=50
ECAL_ENERGY_TRANSFORM=log1p
TPAD_PE_TRANSFORM=log1p
RUN_NAME=slot_2e3e_100k_h192_l3_noise_seed7

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

from ml_ldmx.datasets.ecal_tpad_shards import (
    ShardedECalTpadDataset,
    validate_ml_ready_sharded_cache,
)

root = Path(sys.argv[1]).resolve()
required = int(sys.argv[2])
for label in ("2e", "3e"):
    cache = root / label / "events"
    manifest, index = validate_ml_ready_sharded_cache(cache)
    spec = manifest["cache_spec"]
    count = int(index["num_events"])
    if count < required:
        raise RuntimeError(f"{label} cache has {count:,} events; need at least {required:,}.")
    if spec.get("ecal_energy_transform") != "log1p":
        raise RuntimeError(f"{label} cache does not store log1p ECal energy.")
    if spec.get("tpad_pe_transform") != "log1p":
        raise RuntimeError(f"{label} cache does not store log1p TPad pe.")
    if spec.get("filter_noise") is not False or spec.get("supervise_noise") is not True:
        raise RuntimeError(
            f"{label} cache cannot provide explicit noise targets: "
            f"filter_noise={spec.get('filter_noise')!r}, "
            f"supervise_noise={spec.get('supervise_noise')!r}."
        )
    event = ShardedECalTpadDataset(cache, max_events=1)[0]
    noise_target = event.get("is_noise_target")
    if noise_target is None or noise_target.shape != event["physical_y"].shape:
        raise RuntimeError(f"{label} cache has no aligned is_noise_target field.")
    print(
        f"validated {label}: {count:,} events in {len(index['shards']):,} shards; "
        "explicit noise targets available"
    )
PY
fi

[[ ! -e "${OUTPUT_ROOT}" ]] || fail "Campaign output already exists: ${OUTPUT_ROOT}"
mkdir -p "${REPO_ROOT}/outputs/slurm" "${OUTPUT_ROOT}"

MANIFEST_FILE="${OUTPUT_ROOT}/campaign_manifest.txt"
JOBS_FILE="${OUTPUT_ROOT}/submitted_jobs.tsv"
printf 'job_id\tjob_name\trun_name\ttotal_events\tevents_2e\tevents_3e\n' > "${JOBS_FILE}"
{
  printf 'campaign=%s\n' "${CAMPAIGN}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'git_branch=%s\n' "${GIT_BRANCH}"
  printf 'processed_cache_root=%s\n' "${PROCESSED_CACHE_ROOT}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'events_2e=%s\n' "${EVENTS_PER_SOURCE}"
  printf 'events_3e=%s\n' "${EVENTS_PER_SOURCE}"
  printf 'events_total=%s\n' "${TOTAL_EVENTS}"
  printf 'noise_supervision=%s\n' "${SUPERVISE_NOISE}"
  printf 'epochs_max=%s\n' "${EPOCHS}"
  printf 'early_stopping_min_epochs=%s\n' "${EARLY_STOPPING_MIN_EPOCHS}"
  printf 'early_stopping_patience=%s\n' "${EARLY_STOPPING_PATIENCE}"
  printf 'early_stopping_min_delta=%s\n' "${EARLY_STOPPING_MIN_DELTA}"
  printf 'batch_size=%s\n' "${BATCH_SIZE}"
  printf 'learning_rate=%s\n' "${LR}"
  printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
  printf 'hidden_dim=%s\n' "${HIDDEN_DIM}"
  printf 'num_layers=%s\n' "${NUM_LAYERS}"
  printf 'num_heads=%s\n' "${NUM_HEADS}"
  printf 'dropout=%s\n' "${DROPOUT}"
  printf 'seed=%s\n' "${SEED}"
} > "${MANIFEST_FILE}"

unset RESUME
export_spec="ALL,REPO_ROOT=${REPO_ROOT},VENV_DIR=${VENV_DIR},PROCESSED_CACHE_ROOT=${PROCESSED_CACHE_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},EVENTS_PER_SOURCE=${EVENTS_PER_SOURCE},EPOCHS=${EPOCHS},EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA},BATCH_SIZE=${BATCH_SIZE},LR=${LR},WEIGHT_DECAY=${WEIGHT_DECAY},HIDDEN_DIM=${HIDDEN_DIM},NUM_LAYERS=${NUM_LAYERS},NUM_HEADS=${NUM_HEADS},DROPOUT=${DROPOUT},LAMBDA_ORIGIN=${LAMBDA_ORIGIN},LAMBDA_FRACTION=${LAMBDA_FRACTION},LAMBDA_SLOT=${LAMBDA_SLOT},LAMBDA_COUNT=${LAMBDA_COUNT},GRAD_CLIP=${GRAD_CLIP},LR_SCHEDULER=${LR_SCHEDULER},PLATEAU_PATIENCE=${PLATEAU_PATIENCE},PLATEAU_FACTOR=${PLATEAU_FACTOR},SHARD_CACHE_SIZE=${SHARD_CACHE_SIZE},SEED=${SEED},SUPERVISE_NOISE=${SUPERVISE_NOISE},RUN_PREFLIGHT=${RUN_PREFLIGHT},PREFLIGHT_EVENTS_PER_SOURCE=${PREFLIGHT_EVENTS_PER_SOURCE},ECAL_ENERGY_TRANSFORM=${ECAL_ENERGY_TRANSFORM},TPAD_PE_TRANSFORM=${TPAD_PE_TRANSFORM},RUN_NAME=${RUN_NAME}"
sbatch_args=(
  --parsable
  --chdir="${REPO_ROOT}"
  --job-name=ml_slot_100k
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
  job_id=DRY_RUN
else
  submission="$(sbatch "${sbatch_args[@]}")"
  job_id="${submission%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: ${submission}"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${job_id}" ml_slot_100k "${RUN_NAME}" "${TOTAL_EVENTS}" \
  "${EVENTS_PER_SOURCE}" "${EVENTS_PER_SOURCE}" | tee -a "${JOBS_FILE}"

printf 'campaign=%s\n' "${CAMPAIGN}"
printf 'output_root=%s\n' "${OUTPUT_ROOT}"
printf 'jobs_file=%s\n' "${JOBS_FILE}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'No job was submitted because DRY_RUN=1.\n'
fi
