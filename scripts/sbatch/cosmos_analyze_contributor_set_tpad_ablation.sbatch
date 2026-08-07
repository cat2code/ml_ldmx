#!/bin/bash
#SBATCH -J ml_cs_tpad_ablation
#SBATCH -p lu48
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --no-requeue
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

# Paired inference-only TPad ablation for a trained contributor-set slot model.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed contributor-set run directory}"
CHECKPOINT="${CHECKPOINT:-checkpoints/best.pt}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EVENTS="${MAX_EVENTS:-}"
MAX_PLOT_HITS="${MAX_PLOT_HITS:-100000}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

module --force purge
module load SoftwareTree/Milan
module load GCC/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}"
mkdir -p outputs/slurm
source "${VENV_DIR}/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

echo "host: $(hostname)"
echo "started: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID:-manual} (${SLURM_JOB_NAME:-manual})"
echo "git commit: $(git rev-parse HEAD)"
echo "run directory: ${RUN_DIR}"
echo "checkpoint: ${CHECKPOINT}"
echo "split: ${SPLIT}"
python --version
python -c "import matplotlib, ml_ldmx, numpy, torch; print('TPad-ablation dependencies ready; torch threads:', torch.get_num_threads())"

cmd=(
  python -u scripts/analyze_contributor_set_tpad_ablation.py
  --run-dir "${RUN_DIR}"
  --checkpoint "${CHECKPOINT}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --max-plot-hits "${MAX_PLOT_HITS}"
  --device cpu
)
if [[ -n "${MAX_EVENTS}" ]]; then
  cmd+=(--max-events "${MAX_EVENTS}")
fi
if [[ -n "${PROCESSED_CACHE_ROOT}" ]]; then
  cmd+=(--processed-cache-root "${PROCESSED_CACHE_ROOT}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

printf 'command:'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -v "${cmd[@]}"
else
  "${cmd[@]}"
fi

echo "finished: $(date --iso-8601=seconds)"
