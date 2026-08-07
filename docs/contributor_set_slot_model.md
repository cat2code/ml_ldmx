# Contributor-set slot model

`ECalTpadContributorSetSlotModel` is an experimental, MLPF-inspired model
implemented alongside the legacy `ECalTpadSlotModel`. It does not change the
legacy model, its trainer, its checkpoints, or its Slurm campaign.

## What the model learns

| Level | Head | Target and loss | Purpose |
| --- | --- | --- | --- |
| event | slot validity (3 logits) | prefix slot validity, binary cross-entropy | establishes whether the event contains 2 or 3 electrons |
| ECal hit | contributor set (8 logits) | exact subset of contributing electrons, weighted multiclass cross-entropy | distinguishes noise, pure hits, and every possible mixed-electron combination |
| ECal hit | energy fractions (4 logits) | `[noise, e1, e2, e3]` truth fractions, soft cross-entropy | estimates the energy split among contributors |

There is intentionally no independent electron-count head, hard-origin head,
or mixed/not-mixed head. All three quantities are reconstructed coherently
from the learned heads above.

The three training objectives use equal default coefficients:
`lambda_support=1.0`, `lambda_fraction=1.0`, and `lambda_slot=1.0`.

For three possible electron slots, contributor-set classes use a bit mask:

| Class | Contributors |
| ---: | --- |
| 0 | noise / empty set |
| 1 | e1 |
| 2 | e2 |
| 3 | e1 + e2 |
| 4 | e3 |
| 5 | e1 + e3 |
| 6 | e2 + e3 |
| 7 | e1 + e2 + e3 |

The support target is generated automatically from positive truth fractions.
The training split supplies tempered square-root inverse-frequency weights so
rare mixed-support classes matter without receiving the extreme weights of
full inverse-frequency balancing.

## Coherent reconstruction

Postprocessing is part of the model contract:

1. Slot probabilities are scored against the legal prefixes `{e1,e2}` and
   `{e1,e2,e3}`; the more probable prefix gives the electron count.
2. Contributor sets containing an inactive event slot are assigned zero
   probability.
3. The most probable remaining contributor set is selected for each hit.
4. Fraction logits outside that set are masked, then the remaining fractions
   are normalized to sum to one.
5. Dominant origin is the largest reconstructed fraction. A hit is mixed when
   its selected learned contributor set contains at least two electrons.

The reported mixed-hit probability is the total learned probability of all
multi-electron contributor sets. There is no user-chosen fraction-purity
threshold.

## Training and artifacts

The standalone entry point defaults to 5,000 events from each source (10,000
total), log1p inputs, the full noise-supervised production cache, a 3-layer
pre-LN Transformer, and true padded event batching:

```bash
python scripts/train_ecal_tpad_contributor_set_slot_model.py \
  --processed-cache-root data/processed/production_10M_001_sharded \
  --events-per-source 5000 \
  --epochs 15 \
  --batch-size 8 \
  --device cuda
```

On COSMOS, the dedicated wrapper adds the module/venv setup, a small preflight,
GPU telemetry, and the standard 72-hour ceiling without touching the legacy
slot wrapper:

```bash
sbatch scripts/sbatch/cosmos_train_contributor_set_slot.sbatch
```

The scaling campaign submits 10k CPU first, 50k CPU second, and 100k GPU third.
The jobs are independent—there are no Slurm dependencies—so the scheduler
decides when each starts while the two smaller experiments avoid the occupied
GPU partition:

```bash
bash scripts/submit_cosmos_contributor_set_slot_size_campaign.sh
```

Each run saves:

- `checkpoints/best.pt`, `latest.pt`, and epoch checkpoints;
- `config.json`, including model semantics, class counts/weights, transforms,
  dataset selection, and exact train/validation/test splits;
- `history.json`, `history.csv`, and task-specific learning curves;
- final validation/test metrics and full event count/slot predictions;
- contributor-set, derived-origin, mixed-hit, and electron-count confusion
  matrices;
- mixed-probability score/calibration diagnostics and fraction plots;
- a bounded hit-level prediction sample for plot redesign.

The saved checkpoint plus configuration and split indices are sufficient to
recompute full hit-level outputs and create new plots later. To stop cleanly
after the current epoch, create an empty `STOP_AFTER_EPOCH` file in the run
directory. A resumable checkpoint is also written on `KeyboardInterrupt`.
