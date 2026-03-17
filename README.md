# MLDMX - Machine Learning LDMX
Inspired by MLPF, "but for LDMX". 

### Useful commands that

**Run from the outer directory and let Python find scripts by installing them with pip**

cd mldmx
pip install -e .






# `mldmx/` Directory Overview

```text
mldmx/
├── configs/
│   ├── data/
│   ├── model/
│   └── train/
├── data/
├── figures/
├── notebooks/
├── scripts/
├── src/
│   └── mldmx/
│       ├── datasets/
│       ├── eval/
│       ├── io/
│       ├── models/
│       ├── train/
│       └── viz/
└── tests/
````

---

## Summary

* `data/` → inputs
* `configs/` → settings
* `scripts/` → commands
* `src/mldmx/` → implementation
* `notebooks/` → exploration
* `tests/` → stability
* `figures/` → outputs



## `configs/`

Experiment configuration (no logic).

* `data/` — input schema, detector selection, preprocessing
* `model/` — architecture settings
* `train/` — training hyperparameters

---

## `data/`

Raw inputs and cached datasets.

Examples:

* `.root` files
* reduced samples
* cached processed data

Rule: **files only, no code**

---

## `figures/`

Model-related plots and visuals.

Examples:

* debugging plots
* histograms
* event displays
* thesis-ready figures (if relevant)

---

## `notebooks/`

Interactive exploration and prototyping.

Use for:

* ROOT inspection
* quick tests
* feature exploration

Move stable logic into `src/mldmx/`.

---

## `scripts/`

Runnable entry points.

Examples:

* inspect ROOT
* smoke tests
* training
* evaluation

Rule: thin wrappers that call package code.

---

## `src/mldmx/`

Core Python package (code only).

### `io/`

ROOT reading + schema.

* file access
* tree + branch handling

### `datasets/`

ML input construction.

* tensorization
* padding/masking
* graph building
* dataset classes

### `models/`

Neural network architectures.

* GNNs
* blocks
* heads

### `train/`

Training logic.

* loops
* losses
* metrics
* checkpointing

### `eval/`

Inference + evaluation.

* predictions
* physics metrics

### `viz/`

Lightweight visualization utilities.

---

## `tests/`

Pipeline validation.

Examples:

* reader loads events
* tensor shapes correct
* graph builder valid

---

## Intended Workflow

```text
data/ (.root)
  ↓
io/
  ↓
datasets/
  ↓
models/
  ↓
train/ or eval/
  ↓
viz/ / figures/
```

---
