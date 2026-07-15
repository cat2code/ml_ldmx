# Model Ceiling and Track-Seeded Transformer Study

This document records a controlled follow-up to the 20,000-event Transformer
supervisor demo. The goal was to determine why the hit-origin classifiers
plateau near 82% accuracy for 2-electron overlays and 73% for 3-electron
overlays, then test one architecture that uses TriggerPadTracks more
explicitly.

The study uses one seed and one 20,000-event split per multiplicity. It is a
diagnostic architecture study, not a final model ranking or hyperparameter
scan.

## 1. Questions

The experiment addresses four questions in order:

1. Are predictions mostly correct up to an event-wide permutation of electron
   labels?
2. Is the hard dominant-origin target intrinsically ambiguous for a large
   fraction of hits?
3. Does the existing full self-attention Transformer use its TPad tokens?
4. Can an explicit track-seeded slot pathway improve performance?

The short answer is:

- Label binding and mixed-origin cells explain some errors, but not most of
  the plateau.
- The existing Transformer almost completely ignores TPad tokens.
- The new model makes 3e predictions depend measurably on TPad, but does not
  improve held-out accuracy.
- Shower overlap remains the strongest observed difficulty variable and is
  the best target for the next model design.

## 2. Controlled Setup

Both model families use the same inputs and deterministic split:

```text
2e source: data/ldmx_overlay_events_700k_shards_log1p/2e/events
3e source: data/ldmx_overlay_events_700k_shards_log1p/3e/events
events:    20,000 per model
split:     16,000 train / 3,000 validation / 1,000 test
seed:      7
epochs:    5
batch:     8
targets:   canonical-y
ECal:      log1p input, raw energy retained
TPad:      log1p pe input, raw pe retained
```

All reported checkpoints are `checkpoints/best.pt`, selected by validation
loss. This is epoch 4 for all four runs.

The compared models are:

- `ECalTpadTransformer`: the existing full token self-attention baseline.
- `ECalTpadTrackSeededTransformer`: the prototype implemented for this study.

The prototype has more parameters, so a tie is not a parameter-efficiency
success:

| sample | baseline parameters | track-seeded parameters |
| --- | ---: | ---: |
| 2e | 71,810 | 130,564 |
| 3e | 71,875 | 130,694 |

Training on the local Mac took about 22.5 minutes for the seeded 2e model and
32.7 minutes for the seeded 3e model. The original baseline runs took about 15
and 19 minutes, respectively.

## 3. Diagnostic Definitions

### Ordinary hit accuracy

For event $e$ with $H_e$ supervised ECal hits,

\[
A_e = \frac{1}{H_e}\sum_{h=1}^{H_e}
\mathbb{1}\!\left[\hat y_h = y_h\right].
\]

The reported split hit accuracy sums correct hits before dividing, so events
are weighted by their hit count.

### Permutation-invariant accuracy

For $K$ electron classes, the diagnostic tries every event-wide prediction
label permutation $\pi$ and retains the best result:

\[
A_e^{\mathrm{perm}} =
\max_{\pi\in S_K}
\frac{1}{H_e}\sum_{h=1}^{H_e}
\mathbb{1}\!\left[\pi(\hat y_h)=y_h\right].
\]

The recoverable label-binding gap is

\[
\Delta_e^{\mathrm{perm}} = A_e^{\mathrm{perm}}-A_e.
\]

This is an oracle upper bound, not a usable inference metric, because it uses
truth labels to select the permutation.

### Truth-fraction dominance margin

Let the normalized deposited-energy fractions for a hit be sorted as
$f_{(1)}\geq f_{(2)}\geq\cdots$. Its dominance margin is

\[
m_h=f_{(1)}-f_{(2)}, \qquad 0\leq m_h\leq 1.
\]

A value near zero means the two largest origins contribute similar deposited
energy. A value near one means one origin dominates. Accuracy is accumulated
in fixed margin bins, both by hit count and by preserved raw ECal energy.

### Paired TPad ablation

The same checkpoint and same events are evaluated twice. The second view
removes every TPad token while retaining all ECal hits and targets:

\[
\Delta_{\mathrm{TPad}} =
A_{\mathrm{with\ TPad}}-A_{\mathrm{TPad\ removed}}.
\]

This measures whether predictions depend on TPad context. It does not prove
that the model uses the context optimally.

### Normalized shower separation

The default geometric diagnostic projects all ECal hits into the $xy$ plane.
A hit belongs to every origin with a positive truth contribution, with no
energy or fraction weighting. For shower $i$,

\[
\boldsymbol{\mu}_i = \frac{1}{N_i}\sum_{h\in i}\mathbf{x}_h,
\qquad
s_i = \sqrt{\frac{1}{N_i}\sum_{h\in i}
\left\lVert\mathbf{x}_h-\boldsymbol{\mu}_i\right\rVert^2}.
\]

The pairwise normalized separation is

\[
S_{ij}=\frac{\left\lVert\boldsymbol{\mu}_i-\boldsymbol{\mu}_j\right\rVert}
{\sqrt{s_i^2+s_j^2}},
\]

and the event difficulty coordinate is $S_{\min}=\min_{i<j}S_{ij}$. Smaller
values mean stronger projected overlap. Dominant-origin, first-layer, and
first-three-layer variants remain available in the event records.

## 4. Data Audit

### Available TPad tracks

The complete 20,000-event inputs contain these TPad multiplicities:

| sample | TPad token counts | events missing the expected complete context |
| --- | --- | ---: |
| 2e | 0: 69, 1: 2,720, 2: 17,211 | 13.95% |
| 3e | 0: 12, 1: 844, 2: 7,093, 3: 12,050, 4: 1 | 39.75% |

Missing-track handling is therefore central for 3e, not a rare edge case.
The prototype supplies a learned null-track token when context is incomplete.

### Fraction-target provenance correction

The shards were tensorized with physical fraction columns `[1, 2, 3]`. A 2e
training run uses labels `[1, 2]`. The previous canonicalization path could
mistake the three stored columns for `[background, slot 1, slot 2]` and drop
the first physical-origin fraction column.

The loader now maps fraction columns using the stored `target_label_order` and
preserves the original matrix plus `origin_id_fraction_label_order`. Hard
classification labels were not affected, but the old 2e soft-fraction
diagnostic was invalid. Existing shards do not need to be rebuilt.

### How much hard-target ambiguity exists?

On the complete validation splits:

| sample | mean margin | mixed-origin hits | margin <= 0.10 | margin <= 0.25 |
| --- | ---: | ---: | ---: | ---: |
| 2e | 0.9737 | 5.11% | 0.55% | 1.34% |
| 3e | 0.9566 | 8.28% | 0.94% | 2.27% |

The most ambiguous hits are indeed difficult. In the lowest margin bin
`[0, 0.05)`, baseline hit accuracy is 39.2% for 2e and 38.2% for 3e. For
effectively pure hits near margin 1, it is 83.6% and 74.4%.

The important conclusion is one of scale: ambiguous cells are hard, but fewer
than 1% of hits have margin at most 0.10. They cannot explain an 18% to 27%
overall error rate by themselves.

## 5. Existing Transformer Ceiling Results

### Event-wide label permutations

| sample | ordinary validation accuracy | permutation-invariant accuracy | oracle gain | events with a non-identity optimum |
| --- | ---: | ---: | ---: | ---: |
| 2e | 0.8223 | 0.8286 | +0.62 points | 101 / 3,000 |
| 3e | 0.7270 | 0.7418 | +1.48 points | 297 / 3,000 |

Event-wide class swaps explain a measurable part of the 3e gap, but even the
truth-assisted oracle remains far below perfect accuracy.

### TPad removal

| sample | with TPad | TPad removed | gain from TPad |
| --- | ---: | ---: | ---: |
| 2e baseline | 0.822343 | 0.822342 | +0.0001 percentage points |
| 3e baseline | 0.727028 | 0.726902 | +0.0126 percentage points |

The existing full self-attention Transformer is functionally ECal-only for
this task. Giving it TPad tokens does not mean it learns to use them.

## 6. Track-Seeded Prototype

`ECalTpadTrackSeededTransformer` retains the baseline token encoder, then adds
an object-oriented prediction path:

1. Encode all ECal and TPad tokens with the existing Transformer backbone and
   a detector-type embedding.
2. Instantiate one learned canonical slot per output electron class.
3. Cross-attend each slot to available TPad tokens plus a learned null-track
   token.
4. Cross-attend the resulting slots to all ECal hit embeddings.
5. Score every hit against every event-conditioned slot and add a conventional
   per-hit classifier head.

The final conventional head intentionally lets the prototype fall back to the
baseline behavior. The ablation results reveal how much it does so.

## 7. Model Results

### Best-checkpoint accuracy

| sample | model | validation hit accuracy | test hit accuracy | validation mean energy-weighted accuracy |
| --- | --- | ---: | ---: | ---: |
| 2e | baseline | **0.822343** | 0.820280 | **0.862135** |
| 2e | track-seeded | 0.822069 | **0.821218** | 0.861902 |
| 3e | baseline | **0.727028** | 0.727723 | **0.780706** |
| 3e | track-seeded | 0.725804 | **0.727841** | 0.779173 |

The paired test-set mean-event differences are compatible with zero:

- 2e baseline minus seeded: -0.00091, 95% event-bootstrap interval
  `[-0.00203, 0.00020]`.
- 3e baseline minus seeded: -0.00009, 95% event-bootstrap interval
  `[-0.00129, 0.00114]`.

The seeded model therefore ties the baseline on held-out data. The small test
increases are not evidence of a real improvement.

### Did explicit seeding make the model use TPad?

| sample | model | gain from TPad | events helped / hurt / unchanged |
| --- | --- | ---: | ---: |
| 2e | baseline | +0.0001 points | 89 / 87 / 2,824 |
| 2e | track-seeded | +0.0561 points | 1,079 / 952 / 969 |
| 3e | baseline | +0.0126 points | 379 / 295 / 2,326 |
| 3e | track-seeded | **+0.6571 points** | 1,838 / 874 / 288 |

For the seeded 3e model, TPad contributes +0.713 points when all three tokens
are present and +0.577 points when context is incomplete. This is a successful
mechanistic result: the new prediction path makes TPad matter. It is not an
accuracy result, because total performance remains unchanged.

The seeded permutation-invariant ceilings are also nearly unchanged at 0.8284
for 2e and 0.7406 for 3e. Explicit TPad attention did not solve event-wide
label binding.

## 8. Shower Overlap Remains Dominant

The validation events were divided into eight equal-population bins by the
default all-layer, any-contributor $S_{\min}$:

| sample | lowest-separation bin accuracy | highest-separation bin accuracy |
| --- | ---: | ---: |
| 2e baseline | 0.6148 | 0.9449 |
| 2e track-seeded | 0.6124 | 0.9446 |
| 3e baseline | 0.6039 | 0.8392 |
| 3e track-seeded | 0.6048 | 0.8359 |

The architecture comparison is flat relative to the much larger overlap
effect. The models fail in the same physical regime.

Energy-weighted accuracy is also consistently higher than ordinary hit
accuracy. The models classify energetic deposits better and make a larger
share of mistakes on lower-energy hits.

## 9. Conclusion

This experiment rules out three simple explanations:

- The plateau is not mostly an arbitrary label-permutation problem.
- It is not mostly unavoidable ambiguity in mixed-origin ECal cells.
- It is not solved by adding a track-conditioned cross-attention pathway to
  the existing classifier.

The experiment also establishes two positive findings:

- The original Transformer does not use TPad in a meaningful way.
- A dedicated slot pathway can make 3e predictions depend on TPad, so the
  detector context is learnable and the tensor representation is usable.

The current evidence points toward an assignment and shower-coherence
bottleneck. A larger generic Transformer or broad hyperparameter sweep is not
the best next investment.

## 10. Recommended Next Model

The next controlled prototype should be a genuine iterative object-centric
assignment model, developed from the existing slot/MLPF-lite work:

1. Initialize electron slots from TPad tracks when available and learned null
   slots otherwise.
2. Iteratively update slots from ECal hits and normalize assignments across
   slots, rather than adding an unconstrained per-hit fallback head.
3. Train the assignment with both dominant-origin cross-entropy and the stored
   deposited-energy fraction targets.
4. Match predicted slots to truth showers with a permutation-invariant
   Hungarian objective based on fractions, centroids, or both.
5. Evaluate first on 3e with the same seed-7 20k split, because 3e exposes the
   larger binding gap and showed measurable TPad utility.
6. Require improvement specifically in the lowest shower-separation bins, not
   only in aggregate accuracy.

This experiment combines the two signals found here: object-level label
matching and spatially coherent shower assignment. Only after it beats the
baseline on a held-out split should it receive a multi-seed hyperparameter
study.

## 11. Reproduce the Study

Run from `ml_ldmx/`:

```bash
source ../.venv/bin/activate
```

Train the seeded 2e model:

```bash
python scripts/train_hit_classifier_baseline.py \
  --model ECalTpadTrackSeededTransformer \
  --processed-source 2 2e data/ldmx_overlay_events_700k_shards_log1p/2e/events \
  --events-per-source 20000 \
  --valid-labels 1 2 \
  --ecal-energy-transform log1p \
  --tpad-pe-transform log1p \
  --epochs 5 --batch-size 8 --seed 7 --device mps \
  --cache-model-views --no-progress \
  --num-ecal-plots 2 --num-diagnostic-event-displays 3 \
  --output-root outputs/track_seeded_transformer_20k \
  --run-name track_seeded_2e_20k_seed7
```

Train the seeded 3e model by changing the source and labels:

```bash
python scripts/train_hit_classifier_baseline.py \
  --model ECalTpadTrackSeededTransformer \
  --processed-source 3 3e data/ldmx_overlay_events_700k_shards_log1p/3e/events \
  --events-per-source 20000 \
  --valid-labels 1 2 3 \
  --ecal-energy-transform log1p \
  --tpad-pe-transform log1p \
  --epochs 5 --batch-size 8 --seed 7 --device mps \
  --cache-model-views --no-progress \
  --num-ecal-plots 2 --num-diagnostic-event-displays 3 \
  --output-root outputs/track_seeded_transformer_20k \
  --run-name track_seeded_3e_20k_seed7
```

Build a best-checkpoint inspection bundle for either run:

```bash
python scripts/inspect_hit_classifier_run.py \
  --run-dir outputs/track_seeded_transformer_20k/track_seeded_3e_20k_seed7 \
  --checkpoint best.pt --split val --num-events 9 --device mps
```

Measure the label, ambiguity, and paired TPad ceilings without retraining:

```bash
python scripts/analyze_hit_classifier_ceiling.py \
  --run-dir outputs/track_seeded_transformer_20k/track_seeded_3e_20k_seed7 \
  --checkpoint best.pt --split val --device mps
```

Use the same command with a supervisor-demo baseline run to measure its
ceilings. Compare matched best-checkpoint event records with:

```bash
python scripts/compare_hit_classifier_runs.py \
  --run baseline=outputs/supervisor_demo_transformer_20k/transformer_3e_20k_seed7/inspection/best/val \
  --run track_seeded=outputs/track_seeded_transformer_20k/track_seeded_3e_20k_seed7/inspection/best/val \
  --split val \
  --output-dir outputs/model_ceiling_study/comparison_3e_val
```

Use `--device cuda` on a CUDA host or `--device cpu` without an accelerator.

## 12. Artifact Map

Baseline ceiling plots and tables:

```text
outputs/supervisor_demo_transformer_20k/transformer_2e_20k_seed7/ceiling_analysis/best/val/
outputs/supervisor_demo_transformer_20k/transformer_3e_20k_seed7/ceiling_analysis/best/val/
```

Seeded runs, inspections, and ceiling analyses:

```text
outputs/track_seeded_transformer_20k/track_seeded_2e_20k_seed7/
outputs/track_seeded_transformer_20k/track_seeded_3e_20k_seed7/
```

Matched comparison plots and summaries:

```text
outputs/model_ceiling_study/comparison_2e_val/
outputs/model_ceiling_study/comparison_2e_test/
outputs/model_ceiling_study/comparison_3e_val/
outputs/model_ceiling_study/comparison_3e_test/
```

Useful files to open first:

```bash
open outputs/model_ceiling_study/comparison_3e_val/paired_event_accuracy.png
open outputs/model_ceiling_study/comparison_3e_val/accuracy_difficulty_profiles.png
open outputs/track_seeded_transformer_20k/track_seeded_3e_20k_seed7/ceiling_analysis/best/val/tpad_ablation.png
open outputs/track_seeded_transformer_20k/track_seeded_3e_20k_seed7/inspection/best/val/val_representative_events/val_worst_event_12501_interactive.html
```

Each `ceiling_analysis/best/val/` directory contains:

```text
ceiling_summary.json
ceiling_manifest.json
reference_event_accuracy.json
reference_event_accuracy.csv
tpad_ablated_event_accuracy.json
tpad_ablated_event_accuracy.csv
reference_assignment_ceiling_diagnostics.png
tpad_ablation.png
```

## 13. Implementation Map

- [`src/ml_ldmx/eval/event_diagnostics.py`](src/ml_ldmx/eval/event_diagnostics.py):
  permutation, truth-margin, and detector-context records.
- [`src/ml_ldmx/viz/training.py`](src/ml_ldmx/viz/training.py): assignment-ceiling
  and paired TPad-ablation plots.
- [`scripts/analyze_hit_classifier_ceiling.py`](scripts/analyze_hit_classifier_ceiling.py):
  saved-checkpoint analysis command.
- [`src/ml_ldmx/models/ecal_tpad_track_seeded.py`](src/ml_ldmx/models/ecal_tpad_track_seeded.py):
  track-seeded prototype.
- [`src/ml_ldmx/datasets/ecal_tpad_loading.py`](src/ml_ldmx/datasets/ecal_tpad_loading.py):
  physical fraction-column provenance fix.
- [`tests/test_event_accuracy_diagnostics.py`](tests/test_event_accuracy_diagnostics.py),
  [`tests/test_ceiling_analysis.py`](tests/test_ceiling_analysis.py),
  [`tests/test_track_seeded_transformer.py`](tests/test_track_seeded_transformer.py),
  and [`tests/test_target_fraction_mapping.py`](tests/test_target_fraction_mapping.py):
  focused regression coverage.
