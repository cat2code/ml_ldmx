#!/usr/bin/env bash
#SBATCH --job-name=ml_ldmx_analysis
#SBATCH --partition=gpua100i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/sbatch/cosmos_hit_classifier_analysis.sbatch inspect [inspector arguments...]
  sbatch scripts/sbatch/cosmos_hit_classifier_analysis.sbatch ceiling [ceiling arguments...]
  sbatch scripts/sbatch/cosmos_hit_classifier_analysis.sbatch compare [comparison arguments...]

The arguments after the mode are passed unchanged to the corresponding Python
script. Run each Python script with --help for its complete argument list.

Environment overrides:
  REPO_ROOT          ml_ldmx repository root (default: SLURM_SUBMIT_DIR)
  VENV_DIR           virtualenv directory (default: REPO_ROOT/.venv, then ../.venv)
  PYTHON_BIN          Python command inside the environment (default: python)
  SKIP_MODULE_LOAD   set to 1 when modules are already configured
  CLUSTER_MODULES    space-separated modules to load
EOF
}

MODE="${1:-}"
case "${MODE}" in
  inspect)
    PYTHON_SCRIPT="scripts/inspect_hit_classifier_run.py"
    ;;
  ceiling)
    PYTHON_SCRIPT="scripts/analyze_hit_classifier_ceiling.py"
    ;;
  compare)
    PYTHON_SCRIPT="scripts/compare_hit_classifier_runs.py"
    ;;
  -h|--help|help|"")
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unknown analysis mode '${MODE}'." >&2
    usage >&2
    exit 2
    ;;
esac
shift

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_MODULE_LOAD="${SKIP_MODULE_LOAD:-0}"
CLUSTER_MODULES="${CLUSTER_MODULES:-GCCcore/13.2.0 Python/3.11.5}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${REPO_ROOT}" ]] || fail "Repository directory does not exist: ${REPO_ROOT}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
[[ -f "${REPO_ROOT}/${PYTHON_SCRIPT}" ]] || \
  fail "REPO_ROOT does not point to the ml_ldmx repository: ${REPO_ROOT}"

if [[ "${SKIP_MODULE_LOAD}" != "1" ]]; then
  type module >/dev/null 2>&1 || \
    fail "The module command is unavailable. Set SKIP_MODULE_LOAD=1 if the environment is already configured."
  module purge
  read -r -a module_names <<< "${CLUSTER_MODULES}"
  module load "${module_names[@]}"
fi

if [[ -z "${VENV_DIR:-}" ]]; then
  if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    VENV_DIR="${REPO_ROOT}/.venv"
  elif [[ -f "${REPO_ROOT}/../.venv/bin/activate" ]]; then
    VENV_DIR="$(cd "${REPO_ROOT}/../.venv" && pwd)"
  else
    fail "No virtualenv found. Set VENV_DIR or create ${REPO_ROOT}/.venv."
  fi
fi
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="$(cd "${VENV_DIR}" && pwd)"
elif [[ -f "${REPO_ROOT}/${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="$(cd "${REPO_ROOT}/${VENV_DIR}" && pwd)"
fi
[[ -f "${VENV_DIR}/bin/activate" ]] || fail "Missing virtualenv: ${VENV_DIR}"

cd "${REPO_ROOT}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
  fail "Python executable not found after activating ${VENV_DIR}: ${PYTHON_BIN}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONUNBUFFERED=1

echo "host: ${HOSTNAME:-$(hostname)}"
echo "repository: ${REPO_ROOT}"
echo "virtualenv: ${VENV_DIR}"
echo "analysis mode: ${MODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c \
  "import matplotlib, ml_ldmx, numpy, torch, torch_geometric; print('Analysis dependencies are available')"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "nvidia-smi could not query a GPU on this allocation; continuing."
fi

cmd=("${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" "$@")
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
