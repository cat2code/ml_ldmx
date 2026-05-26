# Maintained MLDMX Model Family: Design Baseline

## Status and Scope

This document defines the intended maintained model family and its shared data
contract. It uses `scripts/train_ecal_tpad_slot_model.py` as the current
end-to-end reference for ROOT loading, tensorization, training, evaluation,
checkpointing, and visualization. Existing simple scripts and MLPF-lite code
remain useful prototypes, but are not the naming or task contract for the
maintained family.

Initial scope:

- Predict ECal hit origin-electron assignments.
- Use canonical ordering along the ECal y-direction for comparisons.
- Preserve original physical origin IDs as provenance.
- Compare ECal-only inputs with ECal plus TriggerPadTracks inputs without
  assuming that TriggerPadTracks improves every architecture.
- Support the advanced slot model's fraction, electron-count, and
  noise/background tasks.

Out of scope for the initial shared contract:

- Event-level `is_signal` prediction.
- ECal layer index, timing, and event total reconstructed energy features.

## Maintained Models

| Maintained name | Event view | Architecture | Initial outputs |
| --- | --- | --- | --- |
| `ECalGravNet` | ECal-only derived view | `GravNetConv` | Hit origin class only |
| `ECalTpadGravNet` | ECal plus TriggerPadTracks | `GravNetConv` | Hit origin class only |
| `ECalTransformer` | ECal-only derived view | Full self-attention | Hit origin class only |
| `ECalTpadTransformer` | ECal plus TriggerPadTracks | Full self-attention | Hit origin class only |
| `ECalTpadSlotModel` | ECal plus TriggerPadTracks | Full self-attention, multi-task heads | Hit origin, energy fraction, electron count, noise/background |

The first four models are baseline classifiers. They must not acquire fraction,
count, noise, or event-level heads as a convenience of sharing training code.
`ECalTpadSlotModel` is the advanced model, not a baseline renamed as one.

## Current Reference Workflow

The current slot-model runner is
`scripts/train_ecal_tpad_slot_model.py`.

1. **Input selection.** The script prefers already processed event tensors in
   `data/processed/ecal_tpad_slot_model/`. Otherwise it loads balanced samples
   from the `2e/events` and `3e/events` ROOT directories beneath
   `data/ldmx_overlay_events_700k/`.
2. **ROOT reading.** `mldmx.io.root_files.find_root_files()` orders ROOT files.
   `mldmx.io.root_reader.iter_ecal_rechits_with_truth_and_triggerpad_context()`
   reads ECal RecHits, ECal sim-hit contribution truth, and
   `TriggerPadTracks_overlay` through one tree handle, optionally in chunks.
   Truth contributions are aligned to RecHits by hit ID.
3. **Tensorization.** `mldmx.datasets.ecal_tpad_loading` calls
   `tensorize_ecal_with_triggerpad_context()` and
   `origin_energy_fraction_targets()`. ECal nodes precede TriggerPadTracks
   nodes in a variable-length event tensor. Loss targets exist for ECal nodes
   only; context nodes are selected away by `ecal_mask`.
4. **Target ordering.** For raw loading, the slot script first requests
   physical-origin targets, attaches the known `2e` or `3e` event count, then
   applies its variable-count canonical slot mapping. Its default
   `canonical-y` mode orders origins by the mean ECal y-position of their
   selected hits.
5. **Dataset policy.** The script constructs a deterministic 80/15/5
   train/validation/test split, derives inverse-frequency origin and count
   weights from the training events, and normalizes continuous columns in `x`
   using training-split statistics only.
6. **Model and training.** `ECalTpadSlotModel` encodes all event tokens and
   produces per-token origin and fraction logits plus event-level
   slot-validity and electron-count logits. `mldmx.train.ecal_tpad_slot_model`
   applies hit losses only at ECal nodes, accumulates event-level metrics, and
   optimizes per small group of events.
7. **Evaluation.** `mldmx.eval.ecal_tpad_slot_model.evaluate()` reuses the
   training loss/metric definitions for validation and test splits and can
   collect event-count prediction records.
8. **Artifacts.** Shared artifact and checkpoint helpers write configuration,
   source/split metadata, JSON/CSV histories, final metrics, predictions, and
   `latest`, `best`, periodic, or interrupted checkpoints under
   `outputs/ecal_tpad_slot_model/<run-name>/`.
9. **Plots.** The workflow writes loss/accuracy histories, hit-origin
   confusion plots, event-count confusion plots, and ECal truth/prediction 3D
   plots through `mldmx.viz`.

`scripts/train_ecal_tpad_mlpf_lite_scaled.py` adds a useful signature-based
tensor cache for a scalable single-source workflow, but its two-head MLPF-lite
task is not one of the maintained targets.

## Canonical Tensor-Event Contract

Future maintained models should consume one canonical mixed-detector event
produced once from ROOT input. The table below is a target schema; it keeps the
working eight-column feature representation but removes ambiguous target
naming that exists in the current code.

### Required Core Fields

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `x` | `float32 [N, 8]` | Mixed-node features using the fixed layout below. |
| `ecal_mask` | `bool [N]` | Nodes supervised as ECal hits. |
| `tpad_mask` | `bool [N]` | TriggerPadTracks context nodes; disjoint from `ecal_mask`. |
| `ecal_pos` | `float32 [N_ecal, 3]` | Selected ECal hit `(x, y, z)` coordinates. |
| `tpad` | `float32 [N_tpad, 2]` | TriggerPadTracks `(centroid, pe)` values. |
| `origin_id_y` | `long [N_ecal]` | Dominant physical origin ID for provenance; `-1` when explicit noise has no contribution truth. |
| `canonical_y` | `long [N_ecal]` | Zero-based electron class ordered by increasing mean ECal y; `-1` for explicit noise rows. |
| `target_label_order` | `list[int]` | Physical origin IDs represented by `canonical_y` classes in order. |
| `event_idx` | `long` | Stable event identifier within the selected tensor dataset. |
| `electron_count` | `long` | Known event electron multiplicity when available or unambiguously derived. |

Feature layout for `x`:

```text
column:  0        1        2       3       4       5            6              7
field:   is_ecal  is_tpad  ecal_x  ecal_y  ecal_z  ecal_energy  tpad_centroid  tpad_pe
```

- ECal nodes populate columns `0, 2:6`; their TPAD columns are zero.
- TriggerPadTracks nodes populate columns `1, 6:8`; their ECal columns are
  zero.
- The current implementation places ECal nodes first and TPAD nodes second.
  Consumers should use masks rather than rely solely on ordering.
- Manifest/config metadata must record schema version, feature layout, noise
  policy, target ordering convention, selected ROOT files, and source entries.

### Optional Advanced and Provenance Fields

| Field | Type and shape | Consumer |
| --- | --- | --- |
| `fraction_target` | `float32 [N_ecal, K]` | Slot model; energy fractions in canonical-y electron order. |
| `origin_id_fraction_target` | `float32 [N_ecal, K]` | Provenance; fractions in original physical-ID order. |
| `is_noise_target` | `bool [N_ecal]` | Slot model only, when noise hits are intentionally retained. |
| `source_file`, `source_entry`, `source_label` | scalar metadata | Reproducibility and diagnosis. |
| `edge_index` | `long [2, E]` | Optional derived graph artifact, never the canonical raw event. |

`K` is the represented maximum number of electron slots for the run. The
canonical event must not use one field name for two meanings. In particular:

- `origin_id_y` always means original physical IDs.
- `canonical_y` always means the comparison/training class ordered along ECal y.
- A background/noise target, if present, is explicit and is not encoded by
  overwriting either provenance field.

The existing code uses `y` and `physical_y` in transitional ways. A migration
adapter may read those fields, but maintained code should expose the explicit
names above before model losses or plots consume the event.

### Targets by Model Family

The four baseline classifiers consume `canonical_y` on ECal hits only. They
ignore advanced optional targets, even when those targets are present in a
cached canonical event.

The slot model consumes the mixed-detector event and may additionally consume
`fraction_target`, `electron_count`, and `is_noise_target`. Noise/background
supervision is an explicit advanced-only training mode:

- The normal slot workflow filters noise hits at training access time; output class `0` consequently
  has no hit-level positive targets by default.
- `--supervise-noise` retains `noise_flag` hits and assigns them hard class
  `0`; this does not change the four baseline comparison paths.
- The slot runner rejects its legacy `--keep-noise` switch because it neither
  assigns background class `0` nor handles noise rows without contribution
  truth.
- Noise hits have `canonical_y=-1`, so canonical-y electron ordering is
  computed only from non-noise hits and keeps its existing slot meaning.
- A noise hit's fraction target is `[1, 0, ..., 0]` at the slot-model loss
  boundary. Its raw electron-origin fraction row is zero because flagged noise
  can have no contribution truth.
- Noise hits do not make an electron slot valid and do not increase
  `electron_count`; count targets remain event electron multiplicity.
- Evaluation reuses the slot-model loss/metric path, so retained noise rows
  appear as truth class `0` in hit accuracy and confusion outputs.
- When a flagged noise hit has no physical-origin contribution,
  `origin_id_y=-1` marks unavailable provenance for that row.

Legacy processed tensor caches do not store `is_noise_target`; explicit noise
supervision cannot use them. New sharded caches store `is_noise_target` and
background-labelled noise rows by default, allowing training runs to either
filter noise at access time or explicitly supervise it without retensorizing
ROOT files.

## ECal-Only Derived View

ECal-only comparisons must be made from the same canonical events, split,
noise policy, and canonical-y labels used for ECal plus TriggerPadTracks
models. No second ROOT reader or alternate tensorization pipeline is needed.

For each canonical event, an ECal-only adapter should derive:

```python
x_ecal = event["x"][event["ecal_mask"], 2:6]  # x, y, z, reconstructed energy
pos = event["ecal_pos"]
target = event["canonical_y"]
origin_id = event["origin_id_y"]
```

`ECalTransformer` receives this variable-length ECal token tensor.
`ECalGravNet` derives its graph/learned-neighborhood input from the same ECal
view and `pos`. The TPAD variants receive full `x` plus masks. Continuous
normalization is fitted using training-split values for the selected view, but
does not change event selection or target ordering.

## Ownership Boundaries

### Already Reusable

| Code | Reuse |
| --- | --- |
| `src/mldmx/io/branches.py`, `root_reader.py`, `root_files.py` | Branch definitions, hit/truth alignment, combined ROOT streaming, file ordering. |
| `src/mldmx/datasets/tensorize.py` | Core ECal/TPAD feature construction and physical contribution fractions. |
| `src/mldmx/datasets/ecal_tpad_dataset.py` | Per-event tensor storage, manifests, and optional PyG conversion. |
| `src/mldmx/datasets/preprocess.py`, `stats.py` | Training-only normalization and basic reporting. |
| `src/mldmx/train/splits.py`, `checkpoints.py`, `logging.py`, `paths.py`, `progress.py`, `batching.py` | Run mechanics used by current workflows. |
| `src/mldmx/io/artifacts.py`, `src/mldmx/viz/` | Artifact writing and useful diagnostic plotting primitives. |

### Extract Later

- One canonical event builder and schema validator, including explicit
  provenance and canonical-y fields.
- One cache/manifest policy that can represent mixed multiplicities, target
  convention, schema version, and noise policy.
- ECal-only and mixed-detector view adapters.
- Common baseline classification loss, metrics, checkpoint metadata, and plot
  labeling.
- A single canonicalization implementation that supports both fixed `3e` and
  variable `2e`/`3e` events.

### Slot-Specific Logic

- `ECalTpadSlotModel` multi-task output heads and event representation.
- Fraction, slot-validity, electron-count, and future explicit noise losses.
- Mixed-multiplicity sampling needed for electron-count supervision.
- Event-count predictions and count confusion plots.

### Prototype or Legacy Code

| Code | Status |
| --- | --- |
| `scripts/simple_3_class_classification_*.py` | Small focused prototypes, not maintained runners. |
| `scripts/root_to_tensor_smoke.py` | Early ROOT-to-padded-tensor smoke test. |
| `scripts/preprocess_ecal_tpad_dataset.py` | Useful prototype preprocessor; lacks the final contract/caching policy. |
| `scripts/preprocess_dataset.py` | Placeholder. |
| `scripts/train_ecal_tpad_mlpf_lite_scaled.py` and `src/mldmx/models/ecal_tpad_mlpf_lite.py` | Current scalable two-head experiment and migration reference, not a maintained target model. |
| `src/mldmx/models/ecal_tpad_gnn.py` | `GraphConv` TPAD prototype; not the maintained GravNet architecture. |
| `src/mldmx/models/gnn_gravnet.py` | Maintained `ECalGravNet` and `ECalTpadGravNet` architectures. |
| `src/mldmx/models/simple_gnn.py`, `src/mldmx/train/train_tiny_gnn.py` | Infrastructure/dummy event-classification prototype. |

`scripts/train_hit_classifier_baseline.py` is the maintained entry point for
the four baseline classifiers and supersedes the simple classification scripts
for comparison experiments without deleting those prototype references.

## Performance Guidance

`scripts/benchmark_common_pipeline.py` measures processed-cache loading, ROOT
read plus tensorization, canonical target preparation, model views, and small
forward/backward steps. Use processed smoke events for quick CPU checks. For
larger repeated training runs, prepare and reuse canonical tensor caches;
ROOT-backed preparation should expose `read_step_size` and be benchmarked on
the cluster filesystem rather than repeated for every experiment.

Model views are immutable after target preparation and feature normalization.
When profiling warrants the additional memory, the maintained baseline trainer
supports `--cache-model-views` to derive its selected ECal-only or
mixed-detector view once per event and reuse it across epochs, avoiding
repeated adapter validation and slicing without changing targets or features.

For large repeated training jobs, the scalable storage path is
`src/mldmx/datasets/ecal_tpad_shards.py`: one numerically ordered input ROOT
file is tensorized into one ML-ready `.pt` shard with a manifest and
event-offset index. Shards store the same canonical combined event contract
and physical-origin provenance, including explicit noise rows by default.
With `--processed-cache`, maintained trainers
validate ROOT-file and tensorization metadata before reuse, create missing
caches once, apply canonical targets and training normalization lazily, and
retain only recently accessed shards in memory. The legacy per-event `.pt`
format remains supported for existing small datasets.

## Current Inconsistencies and Risks

| Risk | Current observation | Required resolution |
| --- | --- | --- |
| Noise/background supervision | `ECalTpadSlotModel` reserves class `0`, while default training filters noise and therefore does not train positive background hits. | Store explicit noise targets in new shards by default; use advanced-only `--supervise-noise` to retain them during training. Noise rows are class `0`, excluded from canonical ordering/count, and receive background-only fraction targets. |
| Label meaning | Canonicalization currently overwrites `physical_y` with canonical slot numbers and stores original IDs in `origin_id_y`; other paths use `physical_y` for physical IDs. | Adopt immutable `origin_id_y` plus explicit `canonical_y`. |
| Variable multiplicity canonicalization | Dataset-level `apply_target_mode()` requires every configured physical label, while the slot runner contains separate logic for mixed `2e`/`3e` samples. | Replace duplicate target mapping with one variable-count canonicalizer. |
| Output contract | MLPF-lite uses three electron outputs; the slot model uses `max_electrons + 1` outputs and prepends an all-zero background fraction column. | Publish separate baseline and slot output schemas with tested conversions. |
| Auxiliary slot-validity output | Current slot training optimizes `slot_valid_logits` in addition to electron-count prediction, while the maintained task list names count rather than a separate validity product. | Decide whether slot validity remains an internal auxiliary loss or a published slot-model output. |
| Out-of-scope signal output | The slot model currently returns `signal_logit` despite no label, loss, or scoped study for `is_signal`. | Remove from the maintained output contract or keep it explicitly disabled during migration. |
| Model identity | Maintained baseline class names now exist; legacy TPAD `GraphConv` code and simple scripts remain alongside them. | Use only the maintained class names and common baseline trainer for comparison runs. |
| Feature dimensions | ECal-only prototypes consume four features, mixed-token models consume eight, and normalization assumes mixed columns beginning at index two. | Centralize derived views and view-aware training-only normalization. |
| Masks and graph artifacts | Mixed models rely on `ecal_mask`; ECal-only paths do not carry it, and PyG conversion currently carries only a subset of advanced targets. | Validate view-specific required fields and keep graph construction derived from canonical events. |
| Terminology | Notes refer to `Trigger Scintillator`; branch names and implementation use `TriggerPadTracks`; code alternates `ECal`, `ECAL`, `TPAD`, and `Tpad`. | Use `ECal` and `TriggerPadTracks` in maintained documentation and APIs; reserve `Tpad` only in agreed class names. |
| Processed input compatibility | The slot runner accepts any event tensor files found in its processed directory without checking that balanced sources, count targets, target mode, or fraction fields match the requested run. | Require manifest/schema compatibility checks before reused tensors enter maintained training. |

## Bounded Refactor Order

1. Freeze the names, terminology, canonical-y convention, noise decision, and
   a versioned tensor-event schema; add contract validation and small tests.
2. Consolidate ROOT-to-canonical-event tensorization and caching, preserving
   physical origin provenance and mixed-multiplicity source metadata.
3. Add ECal-only and mixed-detector view adapters plus shared split,
   normalization, metric, artifact, and classification-runner behavior.
4. Implement and verify the two transformer baseline classifiers against the
   common event contract.
5. Implement and verify the two `GravNetConv` baseline classifiers against the
   same split and targets.
6. Migrate `ECalTpadSlotModel` onto the canonical contract only after
   noise/background labels and multi-task output meanings are explicit.
7. Mark prototypes as historical/migration references in documentation after
   maintained runners reproduce the intended workflow; do not delete them as
   part of this refactor.
