#!/bin/bash
#SBATCH -J ml_ldmx_validate_gpu
#SBATCH -p gpua100i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-data/processed/production_10M_001_sharded}"
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-10}"
MIN_EVENTS_PER_SOURCE="${MIN_EVENTS_PER_SOURCE:-1000000}"
ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM:-log1p}"
TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM:-log1p}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-128}"
SPACE_DIMENSIONS="${SPACE_DIMENSIONS:-4}"
PROPAGATE_DIMENSIONS="${PROPAGATE_DIMENSIONS:-32}"
K="${K:-16}"

module --force purge
module load SoftwareTree/Milan
module load GCC/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}"

# Useful for outputs produced by the script itself.
mkdir -p outputs/slurm

source "${VENV_DIR:-${REPO_ROOT}/.venv}/bin/activate"
python -c "import ml_ldmx, torch, torch_geometric; print('Validation dependencies are available')"

echo "host: $(hostname)"
echo "repo: ${REPO_ROOT}"
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
echo "events per source: ${EVENTS_PER_SOURCE}"
echo "minimum events per source: ${MIN_EVENTS_PER_SOURCE}"
echo "ECal energy transform: ${ECAL_ENERGY_TRANSFORM}"
echo "TPAD PE transform: ${TPAD_PE_TRANSFORM}"
echo "smoke architecture: hidden=${HIDDEN_DIM}, layers=${NUM_LAYERS}, heads=${NUM_HEADS}, feedforward=${DIM_FEEDFORWARD}, space=${SPACE_DIMENSIONS}, propagate=${PROPAGATE_DIMENSIONS}, k=${K}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python --version
nvidia-smi

python -u tests/check_cuda_environment.py

PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT}" \
MIN_EVENTS_PER_SOURCE="${MIN_EVENTS_PER_SOURCE}" \
ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM}" \
TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM}" \
python - <<'PY'
import os
from pathlib import Path

import torch
import torch_cluster
import torch_scatter

from ml_ldmx.datasets.ecal_tpad_shards import validate_ml_ready_sharded_cache

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the COSMOS GPU validation job.")

root = Path(os.environ["PROCESSED_CACHE_ROOT"]).resolve()
minimum = int(os.environ["MIN_EVENTS_PER_SOURCE"])
expected_ecal = os.environ["ECAL_ENERGY_TRANSFORM"]
expected_tpad = os.environ["TPAD_PE_TRANSFORM"]
for label in ("2e", "3e"):
    manifest, index = validate_ml_ready_sharded_cache(root / label / "events")
    count = int(index["num_events"])
    spec = manifest["cache_spec"]
    if count < minimum:
        raise RuntimeError(f"{label} cache has {count} events; need at least {minimum}.")
    if spec.get("ecal_energy_transform") != expected_ecal:
        raise RuntimeError(f"{label} ECal transform does not match {expected_ecal!r}.")
    if spec.get("tpad_pe_transform") != expected_tpad:
        raise RuntimeError(f"{label} TPad transform does not match {expected_tpad!r}.")
    print(f"validated {label} cache: {count} events, {len(index['shards'])} shards")

print("CUDA, torch-cluster, torch-scatter, and production caches are available")
PY

HIDDEN_DIM="${HIDDEN_DIM}" \
NUM_LAYERS="${NUM_LAYERS}" \
NUM_HEADS="${NUM_HEADS}" \
DIM_FEEDFORWARD="${DIM_FEEDFORWARD}" \
SPACE_DIMENSIONS="${SPACE_DIMENSIONS}" \
PROPAGATE_DIMENSIONS="${PROPAGATE_DIMENSIONS}" \
K="${K}" \
python - <<'PY'
import gc
import os

import torch
import torch.nn.functional as F

from ml_ldmx.models import ECalTpadGravNet, ECalTpadTransformer

device = torch.device("cuda")
hidden_dim = int(os.environ["HIDDEN_DIM"])
num_layers = int(os.environ["NUM_LAYERS"])
num_heads = int(os.environ["NUM_HEADS"])
dim_feedforward = int(os.environ["DIM_FEEDFORWARD"])
space_dimensions = int(os.environ["SPACE_DIMENSIONS"])
propagate_dimensions = int(os.environ["PROPAGATE_DIMENSIONS"])
k = int(os.environ["K"])
batch_size = 8
tokens_per_event = 512

for output_dim in (2, 3):
    torch.cuda.reset_peak_memory_stats(device)
    gravnet = ECalTpadGravNet(
        in_dim=8,
        hidden_dim=hidden_dim,
        out_dim=output_dim,
        num_layers=num_layers,
        space_dimensions=space_dimensions,
        propagate_dimensions=propagate_dimensions,
        k=k,
        dropout=0.1,
    ).to(device)
    graph_x = torch.randn(batch_size * tokens_per_event, 8, device=device)
    graph_batch = torch.arange(batch_size, device=device).repeat_interleave(tokens_per_event)
    graph_target = torch.randint(output_dim, (graph_x.shape[0],), device=device)
    graph_logits = gravnet(graph_x, batch=graph_batch)
    F.cross_entropy(graph_logits, graph_target).backward()
    torch.cuda.synchronize()
    print(
        f"GravNet output={output_dim}: {sum(p.numel() for p in gravnet.parameters())} parameters; "
        f"peak CUDA={torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB"
    )
    del gravnet, graph_x, graph_batch, graph_target, graph_logits
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    transformer = ECalTpadTransformer(
        in_dim=8,
        d_model=hidden_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        out_dim=output_dim,
    ).to(device)
    token_x = torch.randn(batch_size, tokens_per_event, 8, device=device)
    token_target = torch.randint(
        output_dim,
        (batch_size, tokens_per_event),
        device=device,
    )
    token_logits = transformer(token_x)
    F.cross_entropy(token_logits.flatten(0, 1), token_target.flatten()).backward()
    torch.cuda.synchronize()
    print(
        f"Transformer output={output_dim}: {sum(p.numel() for p in transformer.parameters())} parameters; "
        f"peak CUDA={torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB"
    )
    del transformer, token_x, token_target, token_logits
    gc.collect()
    torch.cuda.empty_cache()

print("Exact high-capacity CUDA forward/backward checks passed")
PY

ML_LDMX_PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT}" \
ML_LDMX_EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE}" \
ML_LDMX_SMOKE_DEVICE=cuda \
ML_LDMX_ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM}" \
ML_LDMX_TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM}" \
python -m unittest -v tests/test_model_family_smoke.py
