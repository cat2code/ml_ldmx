#!/usr/bin/env bash
#SBATCH --job-name=mldmx_shard_smoke
#SBATCH --output=/projects/hep/fs9/shared/ldmx/users/eliotmp/tensorize_production_5M_001_smoke_%j.out
#SBATCH --error=/projects/hep/fs9/shared/ldmx/users/eliotmp/tensorize_production_5M_001_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

# Small preflight for the production shard job. It reads the first ROOT file
# from each electron-count sample and stores only the first 100 events.
#
# Submit on Cosmos:
#   cd /projects/hep/fs9/shared/ldmx/users/eliotmp
#   sbatch mpetren-msceng-ldmx/mldmx/scripts/slurm/smoke_production_5M_001_sharded.sbatch

WORK_DIR="${WORK_DIR:-/projects/hep/fs9/shared/ldmx/users/eliotmp}"
REPO_DIR="${REPO_DIR:-${WORK_DIR}/mpetren-msceng-ldmx}"
DATASET_DIR="${DATASET_DIR:-${WORK_DIR}/production_5M_001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/mldmx/data/processed/production_5M_001_sharded_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"
READ_STEP_SIZE="${READ_STEP_SIZE:-100}"
MAX_ROOT_FILES="${MAX_ROOT_FILES:-1}"
MAX_EVENTS_PER_ROOT_FILE="${MAX_EVENTS_PER_ROOT_FILE:-100}"

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

if [[ ! -f "${REPO_DIR}/mldmx/scripts/preprocess_ecal_tpad_sharded.py" ]]; then
  echo "Missing preprocessor in repository: ${REPO_DIR}/mldmx/scripts/preprocess_ecal_tpad_sharded.py" >&2
  exit 1
fi

if [[ ! -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  echo "Missing Python virtual environment: ${REPO_DIR}/.venv/bin/activate" >&2
  exit 1
fi

for label in 2e 3e; do
  if [[ ! -d "${DATASET_DIR}/${label}/events" ]]; then
    echo "Missing ROOT input directory: ${DATASET_DIR}/${label}/events" >&2
    exit 1
  fi
done

cd "${REPO_DIR}"
source .venv/bin/activate

echo "Smoke-test tensorization"
echo "Work area:       ${WORK_DIR}"
echo "Repository:      ${REPO_DIR}"
echo "ROOT dataset:    ${DATASET_DIR}"
echo "Shard output:    ${OUTPUT_ROOT}"
echo "ROOT files/type: ${MAX_ROOT_FILES}"
echo "Events/file:     ${MAX_EVENTS_PER_ROOT_FILE}"
echo "Python:          $(command -v "${PYTHON_BIN}")"
echo "Python version:  $("${PYTHON_BIN}" --version 2>&1)"
echo "Started:         $(date)"

"${PYTHON_BIN}" -c "import awkward, torch, uproot; print('Required imports available: awkward, torch, uproot')"

for electron_count in 2 3; do
  label="${electron_count}e"
  input_dir="${DATASET_DIR}/${label}/events"
  output_dir="${OUTPUT_ROOT}/${label}/events"

  echo
  echo "Tensorizing smoke sample ${label}: ${input_dir} -> ${output_dir}"
  "${PYTHON_BIN}" -u mldmx/scripts/preprocess_ecal_tpad_sharded.py \
    --input-root-dir "${input_dir}" \
    --electron-count "${electron_count}" \
    --source-label "${label}" \
    --output-dir "${output_dir}" \
    --max-root-files "${MAX_ROOT_FILES}" \
    --max-events-per-root-file "${MAX_EVENTS_PER_ROOT_FILE}" \
    --read-step-size "${READ_STEP_SIZE}" \
    --skip-existing
done

echo
echo "Smoke test finished: $(date)"
echo "Small processed cache written below ${OUTPUT_ROOT}"
