#!/usr/bin/env bash
#SBATCH --job-name=ml_ldmx_full_shard_check
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail

# Memory/runtime preflight for the production job. It makes one complete shard
# from the first ROOT file in each electron-count sample.
#
# Submit on Cosmos:
#   cd /path/to/work/ml_ldmx
#   mkdir -p outputs/slurm
#   sbatch tests/preflight_full_shards_production_5M_001.sbatch

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
WORK_DIR="${WORK_DIR:-$(dirname "${REPO_DIR}")}"
DATASET_DIR="${DATASET_DIR:-${WORK_DIR}/production_5M_001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/data/processed/production_10M_001_full_shard_preflight}"
PYTHON_BIN="${PYTHON_BIN:-python}"
READ_STEP_SIZE="${READ_STEP_SIZE:-500}"
ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM:-log1p}"
TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM:-log1p}"

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

if [[ ! -f "${REPO_DIR}/scripts/preprocess_ecal_tpad_sharded.py" ]]; then
  echo "Missing preprocessor in repository: ${REPO_DIR}/scripts/preprocess_ecal_tpad_sharded.py" >&2
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

echo "Full-shard tensorization preflight"
echo "Work area:      ${WORK_DIR}"
echo "Repository:     ${REPO_DIR}"
echo "ROOT dataset:   ${DATASET_DIR}"
echo "Shard output:   ${OUTPUT_ROOT}"
echo "ROOT files/type: 1 complete file"
echo "ECal transform: ${ECAL_ENERGY_TRANSFORM}"
echo "TPAD PE transform: ${TPAD_PE_TRANSFORM}"
echo "Python:         $(command -v "${PYTHON_BIN}")"
echo "Python version: $("${PYTHON_BIN}" --version 2>&1)"
echo "Started:        $(date)"

"${PYTHON_BIN}" -c "import awkward, torch, uproot; print('Required imports available: awkward, torch, uproot')"

for electron_count in 2 3; do
  label="${electron_count}e"
  input_dir="${DATASET_DIR}/${label}/events"
  output_dir="${OUTPUT_ROOT}/${label}/events"

  echo
  echo "Tensorizing one complete ${label} ROOT file: ${input_dir} -> ${output_dir}"
  "${PYTHON_BIN}" -u scripts/preprocess_ecal_tpad_sharded.py \
    --input-root-dir "${input_dir}" \
    --electron-count "${electron_count}" \
    --source-label "${label}" \
    --output-dir "${output_dir}" \
    --max-root-files 1 \
    --read-step-size "${READ_STEP_SIZE}" \
    --ecal-energy-transform "${ECAL_ENERGY_TRANSFORM}" \
    --tpad-pe-transform "${TPAD_PE_TRANSFORM}" \
    --skip-existing
done

echo
echo "Full-shard preflight finished: $(date)"
echo "Inspect MaxRSS with sacct before submitting the complete production job."
