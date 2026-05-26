# MLDMX

Machine-learning prototypes for LDMX event reconstruction, currently focused on
assigning ECal RecHits to incoming electrons using ECal information together
with TriggerPadTracks context. The newer pipelines are MLPF-inspired
transformer models that operate on variable-length event tokens.

## Current Focus

The active ECal/TriggerPad workflow reads ROOT events, builds per-event tensor
representations, trains hit-level prediction heads, and saves metrics,
checkpoints, and diagnostic plots.

Each context-aware event uses an 8-column node feature layout:

```text
[is_ecal, is_tpad, ecal_x, ecal_y, ecal_z, ecal_energy, tpad_centroid, tpad_pe]
```

ECal nodes receive supervised targets; TriggerPadTracks nodes provide context.
The default target mode is `canonical-y`, which orders electron targets by
their spatial position in each event instead of relying on arbitrary physical
origin IDs.

Two training entry points are currently useful:

- `scripts/train_ecal_tpad_slot_model.py`: balanced `2e`/`3e` training with
  hit-origin, energy-fraction, slot-validity, and event electron-count heads.
- `scripts/train_ecal_tpad_mlpf_lite_scaled.py`: scalable three-origin
  hit/fraction training on a configurable ROOT subset, with reusable tensor
  caches.

## Setup

Run commands from this `mldmx/` directory unless stated otherwise.

```powershell
cd mldmx
python -m pip install -e .
```

`pyproject.toml` currently installs the local package only; it does not declare
runtime dependencies. The working environment must already provide the
scientific/ML stack used by the scripts, including PyTorch, PyTorch Geometric,
uproot, awkward, NumPy, and Matplotlib.

## Quick Start

Run a forward/backward smoke test of the current slot model:

```powershell
python scripts/smoke_ecal_tpad_slot_model.py --max-events 3 --device cpu
```

The smoke test first looks for processed events in
`data/processed/ecal_tpad_3class_smoke/`; if they are unavailable it reads from
the `2e` and `3e` ROOT directories.

Train the balanced slot model on a small sample:

```powershell
python scripts/train_ecal_tpad_slot_model.py `
  --events-per-class 100 `
  --epochs 20 `
  --device cuda `
  --run-name slot_100_per_class
```

This script reads balanced samples from:

```text
data/ldmx_overlay_events_700k/2e/events/
data/ldmx_overlay_events_700k/3e/events/
```

It uses `data/processed/ecal_tpad_slot_model/` if processed tensor events have
already been created there. Results are written below
`outputs/ecal_tpad_slot_model/<run-name>/`.

Train the MLPF-lite transformer on a scalable `3e` subset:

```powershell
python scripts/train_ecal_tpad_mlpf_lite_scaled.py `
  --max-events 1000 `
  --epochs 20 `
  --device auto `
  --run-name mlpf_lite_1000
```

The scaled script reads `data/ldmx_overlay_events_700k/3e/events/` by default.
It automatically caches preprocessed events below
`data/processed/ecal_tpad_mlpf_lite_scaled/cache_<signature>/` and reuses the
cache when the ROOT inputs and preprocessing settings match. Add
`--force-preprocess` to rebuild a matching cache.

## Data Processing

For a single ROOT file, the preprocessing utility writes per-event `.pt`
tensors and a `manifest.json` file:

```powershell
python scripts/preprocess_ecal_tpad_dataset.py `
  --root-file path/to/events.root `
  --output-dir data/processed/my_dataset `
  --max-events 100
```

By default noise hits are filtered. Use `--keep-noise` to retain them and
`--no-edge-index` when only token tensors, rather than saved graph edges, are
needed.

For a minimal ROOT-to-tensor inspection path:

```powershell
python scripts/root_to_tensor_smoke.py path/to/events.root --stop 10
```

## Training Outputs

Training runs create timestamped directories unless `--run-name` is supplied.
Common artifacts include:

- `config.json`, `history.json`, `history.csv`, and `train.log`
- `checkpoints/latest.pt`, `checkpoints/best.pt`, and periodic checkpoints
- `final_metrics.json`, loss/accuracy histories, and confusion matrices
- ECal truth/prediction plots and fraction diagnostics
- For the slot model, event-count predictions and event-count confusion plots

Resume a compatible run with `--resume path/to/checkpoints/latest.pt`. The
dataset split, label configuration, and target mode must match the checkpoint.

## Project Layout

```text
mldmx/
  data/                         ROOT inputs and processed tensor caches
  outputs/                      Full training run artifacts
  figures/                      Prototype and notebook visualizations
  models/                       Saved weights from earlier simple experiments
  scripts/                      Runnable preprocessing, smoke, and training entry points
  src/mldmx/
    io/                         ROOT reading, branch definitions, and artifact writers
    datasets/                   Tensorization, cached datasets, graph construction, preprocessing
    models/                     Transformer and graph neural-network architectures
    train/                      Losses, metrics, splits, checkpoints, and training loops
    eval/                       Validation and test evaluation for current pipelines
    viz/                        Training, ECal, fraction, and event-level plots
```

Older `simple_3_class_classification_*.py` scripts remain useful as focused
prototypes for ECal-only, ECal/TriggerPad transformer, and graph baselines. The
slot-model and scaled MLPF-lite scripts contain the current end-to-end training
workflows.
