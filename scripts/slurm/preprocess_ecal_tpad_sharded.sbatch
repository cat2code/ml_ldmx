#!/usr/bin/env bash
#SBATCH --job-name=mldmx_tensorize
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Submit once per independent physical dataset:
# sbatch --export=ALL,SOURCE_LABEL=2e,ELECTRON_COUNT=2,ROOT_DIR=/path/to/2e/events,OUTPUT_DIR=/scratch/$USER/mldmx/ecal_tpad_2e_sharded scripts/slurm/preprocess_ecal_tpad_sharded.sbatch
# sbatch --export=ALL,SOURCE_LABEL=3e,ELECTRON_COUNT=3,ROOT_DIR=/path/to/3e/events,OUTPUT_DIR=/scratch/$USER/mldmx/ecal_tpad_3e_sharded scripts/slurm/preprocess_ecal_tpad_sharded.sbatch

: "${SOURCE_LABEL:?Set SOURCE_LABEL, for example 2e or 3e}"
: "${ELECTRON_COUNT:?Set ELECTRON_COUNT, for example 2 or 3}"
: "${ROOT_DIR:?Set ROOT_DIR to the source ROOT events directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the processed shard cache directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
READ_STEP_SIZE="${READ_STEP_SIZE:-500}"

mkdir -p outputs/slurm

"${PYTHON_BIN}" scripts/preprocess_ecal_tpad_sharded.py \
  --input-root-dir "${ROOT_DIR}" \
  --electron-count "${ELECTRON_COUNT}" \
  --source-label "${SOURCE_LABEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --read-step-size "${READ_STEP_SIZE}" \
  --skip-existing
