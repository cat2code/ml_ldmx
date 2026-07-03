This is a historical overhaul prompt log. Some smoke scripts referenced below were later consolidated into `ml_ldmx/tests`.

Below is the final prompt sequence. Run each prompt in a separate Codex task, in order. Do not give Codex the whole sequence at once; each part is intentionally bounded so that the existing working slot-model workflow stays usable throughout the overhaul.

## Part 1: Document The Reference Architecture And Workflow

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Create the design baseline for the maintained ml_ldmx model family. Treat `scripts/train_ecal_tpad_slot_model.py` as the best current end-to-end reference workflow for IO, ROOT reading, tensorization, training, evaluation, checkpoints, and visualization.

Maintained target models:
1. `ECalGravNet`: ECal-only GravNetConv hit-origin classifier.
2. `ECalTpadGravNet`: ECal + TriggerPad GravNetConv hit-origin classifier.
3. `ECalTransformer`: ECal-only full-self-attention hit-origin classifier.
4. `ECalTpadTransformer`: ECal + TriggerPad full-self-attention hit-origin classifier.
5. `ECalTpadSlotModel`: advanced ECal + TriggerPad multi-task model for origin assignment, energy fraction, electron count, and noise/background.

Scientific constraints:
- The four baseline models perform origin-electron hit classification only.
- Comparing ECal-only versus ECal + TriggerPad inputs is intentional; do not assume TPAD improves every model.
- Canonical ordering along the ECal y-direction is the intended comparison convention, while original physical origin IDs must remain available for provenance.
- Event-level `is_signal` prediction is out of scope.
- Layer index, ECal timing, and total reconstructed energy are possible future studies only, not part of the initial shared contract.

Inspect:
- `notes.md`
- `README.md`
- `scripts/train_ecal_tpad_slot_model.py`
- `scripts/train_ecal_tpad_mlpf_lite_scaled.py`
- `src/ml_ldmx/io/`
- `src/ml_ldmx/datasets/`
- `src/ml_ldmx/models/`
- `src/ml_ldmx/train/`
- `src/ml_ldmx/eval/`
- `src/ml_ldmx/viz/`
- existing prototype scripts

Task:
1. Create a concise design document under `ml_ldmx/docs/`.
2. Describe the current slot-model workflow from ROOT files through tensorization, training, evaluation, checkpoints, and plots.
3. Define the canonical tensor-event schema that future maintained models should consume.
4. Explain how ECal-only models should use a derived view of the same canonical event instead of a separate IO/tensorization pipeline.
5. Identify which code is already reusable, which logic should later be extracted, which logic is slot-specific, and which files are legacy/prototypes.
6. Identify current inconsistencies in model naming, labels, feature dimensions, masks, outputs, noise handling, and terminology such as Trigger Scintillator versus TriggerPadTracks.
7. Propose a bounded implementation order for the later refactor.

Constraints:
- Documentation task only, except for tiny docstring/comment corrections if genuinely needed.
- Do not implement models.
- Do not delete or rename legacy code.
- Do not modify generated data, outputs, checkpoints, figures, or saved weights.
- Respect unrelated working-tree changes.

Verification:
Report the document created, the canonical event contract, the intended maintained model names, and the highest-risk inconsistencies found.
```

## Part 2: Extract Shared Data Loading From The Working Slot Pipeline

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Extract reusable data-loading and event-preparation behavior from `scripts/train_ecal_tpad_slot_model.py`, while keeping the existing slot-model training workflow functional.

Reference workflow:
- `scripts/train_ecal_tpad_slot_model.py`

Relevant modules:
- `src/ml_ldmx/io/root_reader.py`
- `src/ml_ldmx/io/root_files.py`
- `src/ml_ldmx/datasets/tensorize.py`
- `src/ml_ldmx/datasets/ecal_tpad_loading.py`
- `src/ml_ldmx/datasets/ecal_tpad_dataset.py`
- `src/ml_ldmx/datasets/preprocess.py`
- `src/ml_ldmx/train/paths.py`
- `src/ml_ldmx/train/splits.py`

Required design:
- The canonical event representation remains the combined ECal + TriggerPad tensor event currently used by the slot workflow.
- All five maintained models will eventually receive events originating from this common loading/tensorization path.
- ECal-only model support will later be implemented as a view of this canonical event, not a separate ROOT pipeline.
- Preserve original physical origin IDs as provenance while supporting canonical-y target labels for maintained comparisons.

Task:
1. Inspect helper logic currently embedded in `train_ecal_tpad_slot_model.py`.
2. Extract only clearly reusable behavior for:
   - resolving processed versus ROOT-backed inputs;
   - loading processed tensor events;
   - loading ROOT events through existing readers/tensorization;
   - attaching source metadata;
   - target-mode application where generally reusable;
   - deterministic split preparation if it currently belongs outside the script.
3. Place extracted functionality in appropriate existing modules or small narrowly scoped new modules.
4. Update `train_ecal_tpad_slot_model.py` to call the shared functionality without changing its intended behavior or CLI.
5. Preserve the existing processed event format and cache behavior.
6. Report whether default noise filtering means the advanced model is currently unable to train a real per-hit noise/background task; do not solve that issue in this part.

Constraints:
- Do not implement baseline models.
- Do not redesign ROOT branch reading.
- Do not create a second cache schema.
- Do not modify generated artifacts or old run outputs.

Verification:
- Run the existing slot-model CPU smoke path.
- If feasible using existing small data, run a minimal slot training invocation.
- Report extracted functions, remaining slot-specific logic, and validation results.
```

## Part 3: Add Shared Model Input Views

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Add model-facing adapters that derive all required maintained-model inputs from one canonical combined tensor event.

Canonical combined feature layout:
[is_ecal, is_tpad, ecal_x, ecal_y, ecal_z, ecal_energy, tpad_centroid, tpad_pe]

Relevant modules:
- `src/ml_ldmx/datasets/tensorize.py`
- `src/ml_ldmx/datasets/ecal_tpad_dataset.py`
- `src/ml_ldmx/datasets/graph_builder.py`
- shared loading/preparation utilities extracted previously

Task:
1. Add a small explicit adapter module in `src/ml_ldmx/datasets/`.
2. Given one canonical event, support:
   - ECal-only token input for `ECalTransformer`;
   - full ECal + TriggerPad token input for `ECalTpadTransformer`;
   - ECal-only node input for `ECalGravNet`;
   - full ECal + TriggerPad node input for `ECalTpadGravNet`;
   - unchanged full combined input access for `ECalTpadSlotModel`.
3. Ensure the adapters retain or expose:
   - ECal mask;
   - TPAD mask where applicable;
   - ECal positions;
   - hit targets;
   - physical-origin provenance;
   - canonical-y targets where already present;
   - event/source metadata.
4. For ECal-only views, use the physical ECal feature columns derived from the canonical event rather than returning type/TPAD placeholder columns unless there is a documented reason.
5. Validate required keys, feature shapes, mask alignment, and target alignment with clear exceptions.
6. Add a lightweight CPU validation script that loads a saved smoke event through the shared loading path and derives all required views from that exact event.

Constraints:
- Do not implement model architectures yet.
- Do not introduce a new ROOT reader, cache representation, or preprocessing pipeline.
- Do not change slot-model scientific behavior.

Verification:
Run the adapter validation using `data/processed/ecal_tpad_3class_smoke/` and report the shape and target/mask fields of each derived view.
```

## Part 4: Extract Reusable Runtime And Artifact Utilities

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Extract the non-model-specific runtime and artifact-management behavior from the working slot trainer so later baseline trainers can reuse it without duplicating infrastructure.

Reference:
- `scripts/train_ecal_tpad_slot_model.py`

Relevant modules:
- `src/ml_ldmx/io/artifacts.py`
- `src/ml_ldmx/train/checkpoints.py`
- `src/ml_ldmx/train/logging.py`
- `src/ml_ldmx/train/paths.py`
- `src/ml_ldmx/train/progress.py`
- `src/ml_ldmx/viz/`

Task:
1. Identify logic in the slot training script that is genuinely general:
   - device selection and fallback;
   - run directory resolution;
   - logging initialization;
   - configuration/history/final-metrics writing;
   - checkpoint helper patterns where compatible;
   - common ECal hit truth/prediction visualization support.
2. Extract only reusable portions into existing modules or small focused modules.
3. Update `scripts/train_ecal_tpad_slot_model.py` to use the extracted helpers.
4. Keep the following slot-specific:
   - fraction metrics and plots;
   - slot-validity metrics;
   - electron-count metrics and confusion plots;
   - multi-task loss composition.
5. Do not invent a large trainer framework or registry abstraction in this task.

Constraints:
- Preserve intended slot training behavior and command-line arguments.
- Do not implement baseline training yet.
- Do not rewrite checkpoints unless compatibility requires a narrowly documented change.
- Do not modify old output directories.

Verification:
Run the slot-model CPU smoke test and a tiny training invocation if feasible. Confirm normal config/history/checkpoint/final-metrics generation still works.
```

## Part 5: Implement The Two Maintained GravNet Models

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Implement the two maintained GravNetConv baseline architectures using data from the shared slot-derived event pipeline and adapter layer.

Target classes:
1. `ECalGravNet`: ECal-only origin-electron hit classifier.
2. `ECalTpadGravNet`: ECal + TriggerPad context origin-electron hit classifier.

Relevant files:
- `src/ml_ldmx/models/gnn_gravnet.py`
- `src/ml_ldmx/models/__init__.py`
- shared event-view adapters
- `src/ml_ldmx/models/ecal_tpad_gnn.py` as legacy reference only

Scientific intent:
- Both models produce per-node origin-class logits.
- The context-aware model accepts TPAD nodes as context, but only ECal node outputs receive hit-classification supervision.
- TPAD usefulness is an experimental comparison, not an assumption.

Task:
1. Replace the placeholder implementation in `gnn_gravnet.py` with both maintained model classes using `torch_geometric.nn.GravNetConv`.
2. Use a consistent constructor and forward/output contract for both classes.
3. Do not require the older manually built `edge_index` interface if it is not part of the GravNetConv model contract.
4. Export both new classes from `src/ml_ldmx/models/__init__.py`.
5. Preserve `ecal_tpad_gnn.py` as legacy code for now.
6. Add a focused CPU smoke script or test that:
   - loads a canonical event through the shared loader;
   - derives the two GravNet views;
   - runs each model forward;
   - computes appropriate ECal hit cross-entropy loss;
   - checks finite loss;
   - runs backward successfully.

Constraints:
- Do not build full training loops in this task.
- Do not delete legacy models or scripts.
- Do not alter slot-model losses or outputs.

Verification:
Run the new GravNet smoke validation on CPU and report input shapes, logits shapes, supervised-node counts, and finite-loss/backward success.
```

## Part 6: Implement The Two Maintained Transformer Baselines

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Implement the two maintained full-self-attention baseline transformer architectures using the same shared canonical event pipeline as the slot model.

Target classes:
1. `ECalTransformer`: ECal-only origin-electron hit classifier.
2. `ECalTpadTransformer`: ECal + TriggerPad origin-electron hit classifier.

Relevant files:
- `src/ml_ldmx/models/ecal_transformer.py`
- `src/ml_ldmx/models/__init__.py`
- shared event-view adapters
- existing simple transformer scripts as prototype references only

Scientific intent:
- The ECal-only model attends over ECal hit tokens.
- The ECal + TriggerPad model attends jointly over ECal and TPAD tokens.
- Only ECal nodes are supervised in the context-aware baseline.
- Full self-attention is appropriate for the current relatively small event sizes.

Task:
1. Refactor or extend `ecal_transformer.py` so both maintained classes exist with explicit names.
2. Keep a consistent baseline output contract: per-input-node origin-class logits.
3. Ensure `ECalTpadTransformer` performs attention over the full combined token sequence.
4. Preserve a compatibility alias for existing prototype usage only if it prevents unnecessary breakage.
5. Export both maintained classes from `src/ml_ldmx/models/__init__.py`.
6. Add a focused CPU smoke script or test that:
   - loads one canonical event through the shared pipeline;
   - derives ECal-only and ECal + TriggerPad transformer views;
   - runs forward for each model;
   - computes masked classification loss where appropriate;
   - checks finite loss;
   - runs backward successfully.

Constraints:
- Do not add training workflows yet.
- Do not modify the advanced slot model in this task.
- Do not delete legacy scripts.

Verification:
Run the transformer smoke validation on CPU and report model input shapes, logits shapes, supervised-node counts, and successful backward passes.
```

## Part 7: Validate All Five Maintained Models Through One Common Pipeline

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Create one integration validation script showing that all five maintained architectures consume events produced through the shared workflow derived from `scripts/train_ecal_tpad_slot_model.py`.

Maintained models:
1. `ECalGravNet`
2. `ECalTpadGravNet`
3. `ECalTransformer`
4. `ECalTpadTransformer`
5. `ECalTpadSlotModel`

Required target convention:
- Use canonical-y targets for the maintained comparison validation.
- Confirm original physical-origin information remains present for provenance.

Task:
1. Add a script such as `scripts/validate_model_family_common_pipeline.py`.
2. Load canonical events using the shared data-loading/preparation utilities.
3. Prefer `data/processed/ecal_tpad_3class_smoke/` for the default CPU validation path.
4. Support ROOT fallback only if it fits cleanly through the same shared loader.
5. Derive architecture-specific views using the shared adapter layer.
6. For each of the four baseline models:
   - instantiate a small CPU-friendly configuration;
   - run forward;
   - compute ECal hit-origin classification loss;
   - assert finite loss;
   - run backward.
7. For `ECalTpadSlotModel`:
   - use its existing multi-task loss computation;
   - assert relevant losses are finite;
   - run backward.
8. Print a compact result table with model name, input shape, supervised output shape, target mode, and loss.
9. Explicitly report whether the loaded validation event includes noise hits. If noise hits are filtered, state that noise/background training is not validated by this script.
10. Add a short README section documenting how to run this validation.

Constraints:
- This is validation only, not training.
- Do not save checkpoints, plots, or run directories.
- Do not include `ECalTpadMLPFLiteTransformer` among the required five maintained models.
- Do not remove legacy experimental code.

Verification:
Run the new validation script on CPU using the processed smoke dataset and report pass/fail results for all five models.
```

## Part 8: Resolve Noise Supervision For The Advanced Model

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Resolve whether and how `ECalTpadSlotModel` can learn its intended noise/background task, given that the current workflow usually filters noise hits.

Intended advanced-model scope:
ECal + TriggerPad -> full-attention multi-task model -> origin assignment + energy fraction + number of electrons + noise/background.

Explicitly out of scope:
- event-level `is_signal` classification.

Relevant files:
- `src/ml_ldmx/datasets/tensorize.py`
- `src/ml_ldmx/datasets/ecal_tpad_loading.py`
- `src/ml_ldmx/models/ecal_tpad_slot_model.py`
- `src/ml_ldmx/train/ecal_tpad_slot_model.py`
- `src/ml_ldmx/eval/ecal_tpad_slot_model.py`
- `scripts/train_ecal_tpad_slot_model.py`
- `scripts/smoke_ecal_tpad_slot_model.py`

Task:
1. Trace how noise flags are read, filtered, tensorized, represented as targets, used in loss calculation, and evaluated.
2. Determine whether the current default workflow actually trains the background/noise output class or merely includes an unused output class.
3. Document the finding in the appropriate design documentation or README section.
4. If noise supervision is absent because noise hits are filtered, implement the smallest explicit opt-in path for a noise-inclusive advanced-model experiment.
5. Preserve default behavior for the four baseline comparison models unless an explicit option is selected.
6. Ensure canonical-y electron-origin semantics remain correct for non-noise hits.
7. Determine how fraction targets and count targets behave for noise nodes; implement a safe policy only if it is unambiguous from existing code/data, otherwise document the blocking target-policy decision.
8. Add a focused CPU validation for the implemented noise-inclusive path if feasible.

Constraints:
- Do not add signal classification.
- Do not make silent scientific-label changes.
- Do not force noise-inclusive training into baseline models.
- Keep changes narrowly scoped to the advanced task.

Verification:
Report whether noise classification was previously trained, what explicit behavior now exists, and the result of any CPU validation.
```

## Part 9: Add Shared Baseline Training

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Add a common training entry point for the four baseline hit-origin classifiers, reusing infrastructure extracted from the working slot-model workflow.

Baseline models:
1. `ECalGravNet`
2. `ECalTpadGravNet`
3. `ECalTransformer`
4. `ECalTpadTransformer`

Reference workflow:
- `scripts/train_ecal_tpad_slot_model.py`

Task:
1. Create one baseline training script, such as `scripts/train_hit_classifier_baseline.py`, with a `--model` option selecting one of the four baseline models.
2. Reuse existing shared components for:
   - canonical event loading;
   - ROOT/processed selection;
   - source metadata;
   - canonical-y targets and physical-origin provenance;
   - feature normalization where appropriate;
   - deterministic splits;
   - device resolution;
   - run directories and logging;
   - configuration/history/final metric artifacts;
   - checkpoints where compatible;
   - ECal truth/prediction plots.
3. Use baseline-appropriate training only:
   - ECal hit-origin cross-entropy loss;
   - hit accuracy;
   - classification confusion matrix;
   - representative ECal truth/prediction visualizations.
4. Support ECal-only and ECal + TriggerPad input views through the shared adapter layer.
5. Keep `scripts/train_ecal_tpad_slot_model.py` as the specialized advanced-model trainer.
6. Document that the new baseline trainer supersedes the older simple prototype scripts for maintained experiments, without deleting those prototypes.

Constraints:
- Do not duplicate the common data pipeline inside the new script.
- Do not add fraction/count/slot losses to baseline models.
- Do not start substantial training jobs.
- Preserve existing advanced slot workflow.

Verification:
- Run a one-epoch CPU smoke training invocation for each of the four baseline models on a tiny processed dataset.
- Re-run the five-model common-pipeline validation.
- Re-run the existing slot-model smoke validation.
- Report generated smoke artifacts and outcomes.
```

## Part 10: Benchmark And Improve Throughput Before Cluster Deployment

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Measure and improve the performance bottlenecks that matter before moving training to a larger cluster dataset.

Context:
`notes.md` states that ROOT reading and training speed are critical concerns. The common workflow should now be based on the slot-model pipeline and shared across all maintained architectures.

Task:
1. Add a lightweight benchmark script, such as `scripts/benchmark_common_pipeline.py`.
2. Measure, where applicable:
   - ROOT read plus tensorization throughput;
   - processed-cache event loading throughput;
   - adapter/view preparation throughput;
   - forward/backward throughput for small representative configurations of the maintained models.
3. Use small local inputs and CPU by default; use GPU only if already available and straightforward.
4. Record baseline timings before making performance changes.
5. Identify the largest measured bottleneck.
6. Implement only low-risk improvements that preserve data and target semantics, such as:
   - improved reuse of processed caches;
   - avoiding repeated tensor/view conversion;
   - exposing useful ROOT chunk/read settings consistently;
   - eliminating redundant preprocessing in active entry points.
7. Document recommended settings for smoke validation versus larger training jobs.

Constraints:
- Do not change scientific model behavior for speed.
- Do not launch large training jobs.
- Do not optimize unmeasured areas speculatively.
- Do not modify generated historical output runs.

Verification:
Run the benchmark locally and report timings. For any implemented improvement, provide before/after results and remaining bottlenecks.
```

## Part 11: Prepare The Validated Workflow For Cluster Execution

```text
Work inside the `ml_ldmx/` directory only.

Goal:
Prepare the validated common data/model workflow for execution on a computing cluster with the same Python environment and larger available datasets.

Active workflows to support:
1. Five-model common-pipeline validation.
2. Four baseline model training through the shared baseline trainer.
3. Advanced multi-task training through `scripts/train_ecal_tpad_slot_model.py`.

Task:
1. Inspect active scripts for hard-coded local assumptions about:
   - ROOT input locations;
   - processed caches;
   - output/run directories;
   - checkpoints;
   - devices;
   - event limits;
   - random seeds.
2. Ensure the active command-line entry points consistently support explicit:
   - input data root or processed dataset path;
   - cache path where relevant;
   - output/run directory;
   - model selection for baseline training;
   - device;
   - event limit;
   - random seed;
   - resume checkpoint wherever resume is already supported.
3. Preserve local defaults where reasonable.
4. Add a concise cluster execution guide under `ml_ldmx/docs/` showing:
   - package installation in the existing Python environment;
   - five-model validation invocation;
   - example baseline training invocations;
   - example advanced slot-model training invocation;
   - recommended separation of input data, processed caches, outputs, and checkpoints;
   - benchmark guidance from the performance task.
5. State clearly which cluster-specific details are still needed before scheduler submission scripts can be written, such as scheduler type, GPU availability, time limits, memory limits, filesystem paths, and module/environment activation details.

Constraints:
- Do not embed cluster-specific absolute paths into Python modules.
- Do not copy datasets into the repository.
- Do not assume Slurm, PBS, or another scheduler without confirmation.
- Do not run large training jobs.

Verification:
Run tiny local CPU invocations using explicit path and output arguments for the active validation/training entry points. Report success and remaining cluster information needed.
```

## Execution Order

Use the prompts in this order:

1. Document the reference workflow and architecture.
2. Extract shared loading/preparation from the working slot trainer.
3. Add shared model input views.
4. Extract reusable runtime/artifact utilities.
5. Implement GravNet models.
6. Implement transformer baseline models.
7. Validate all five models through the common pipeline.
8. Resolve advanced-model noise supervision.
9. Add common baseline training.
10. Benchmark and optimize measured throughput.
11. Prepare cluster execution.

The key principle for every part is that `train_ecal_tpad_slot_model.py` is the proven workflow foundation. The overhaul should extract and reuse its successful infrastructure, while keeping the model-specific differences scientifically explicit.