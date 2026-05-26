#!/usr/bin/env bash
#SBATCH --job-name=mldmx_tensorize_5M
#SBATCH --output=/projects/hep/fs9/shared/ldmx/users/eliotmp/tensorize_production_5M_001_%j.out
#SBATCH --error=/projects/hep/fs9/shared/ldmx/users/eliotmp/tensorize_production_5M_001_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Expected Cosmos layout:
#   /projects/hep/fs9/shared/ldmx/users/eliotmp/production_5M_001/{2e,3e}/events/*.root
#   /projects/hep/fs9/shared/ldmx/users/eliotmp/mpetren-msceng-ldmx/
#
# Submit after pulling the repository on Cosmos:
#   cd /projects/hep/fs9/shared/ldmx/users/eliotmp
#   sbatch mpetren-msceng-ldmx/mldmx/scripts/slurm/preprocess_production_5M_001_sharded.sbatch
#
# Follow output after submission, replacing JOB_ID with the ID printed by sbatch:
#   tail -F tensorize_production_5M_001_JOB_ID.out tensorize_production_5M_001_JOB_ID.err
#
# The job is resumable: re-submitting it reuses already completed valid shards.

WORK_DIR="${WORK_DIR:-/projects/hep/fs9/shared/ldmx/users/eliotmp}"
REPO_DIR="${REPO_DIR:-${WORK_DIR}/mpetren-msceng-ldmx}"
DATASET_DIR="${DATASET_DIR:-${WORK_DIR}/production_5M_001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/mldmx/data/processed/production_5M_001_sharded}"
PYTHON_BIN="${PYTHON_BIN:-python}"
READ_STEP_SIZE="${READ_STEP_SIZE:-500}"

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

echo "Work area:     ${WORK_DIR}"
echo "Repository:    ${REPO_DIR}"
echo "ROOT dataset:  ${DATASET_DIR}"
echo "Shard output:  ${OUTPUT_ROOT}"
echo "Python:        $(command -v "${PYTHON_BIN}")"
echo "Python version: $("${PYTHON_BIN}" --version 2>&1)"
echo "Started:       $(date)"

"${PYTHON_BIN}" -c "import awkward, torch, uproot; print('Required imports available: awkward, torch, uproot')"

for electron_count in 2 3; do
  label="${electron_count}e"
  input_dir="${DATASET_DIR}/${label}/events"
  output_dir="${OUTPUT_ROOT}/${label}/events"

  echo
  echo "Tensorizing ${label}: ${input_dir} -> ${output_dir}"
  "${PYTHON_BIN}" -u mldmx/scripts/preprocess_ecal_tpad_sharded.py \
    --input-root-dir "${input_dir}" \
    --electron-count "${electron_count}" \
    --source-label "${label}" \
    --output-dir "${output_dir}" \
    --read-step-size "${READ_STEP_SIZE}" \
    --skip-existing
done

echo
echo "Finished:      $(date)"
echo "Processed data written below ${OUTPUT_ROOT}"
