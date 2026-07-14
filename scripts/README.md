# ml_ldmx/scripts

This directory contains runnable Python and sbatch entry points for tasks the
user wants to perform: preprocessing, training, benchmarking, environment
checks, and cluster launch jobs.

Scripts should call reusable functionality from `ml_ldmx/src/ml_ldmx`; they
should not become the place where maintained features are implemented.

## Tensorize the 5M 2e and 3e ROOT datasets on Slurm

The input directories are supplied at submission time. Each ROOT file becomes
one `.pt` shard, and at most `MAX_PARALLEL_JOBS` files run concurrently.

```bash
cd /cluster/path/to/ml_ldmx
module load GCCcore/13.2.0 Python/3.11.5
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
mkdir -p outputs/slurm

sbatch \
  --export=ALL,DATASET_DIR=/cluster/data/production_5M_001,OUTPUT_ROOT=/cluster/data/production_5M_001_shards,MAX_PARALLEL_JOBS=100 \
  scripts/preprocess_production_5M_001_sharded.sbatch
```

`DATASET_DIR` must contain `2e/events/*.root` and `3e/events/*.root`. For any
other layout, replace it with `ROOT_2E_DIR=/path/to/2e` and
`ROOT_3E_DIR=/path/to/3e` in the `--export` list. Set
`MAX_PARALLEL_JOBS` to the largest concurrent job count allowed by the cluster.

```bash
squeue -u "$USER"
tail -F outputs/slurm/ml_ldmx_tensorize_JOB_ID.out
```

The dispatcher runs the test suite and a real-ROOT smoke test, submits the
worker array, then submits a dependent finalizer. The finalizer requires exactly
5,000,000 events for each class and writes:

```text
/cluster/data/production_5M_001_shards/2e/events/{manifest.json,index.json,shards/*.pt}
/cluster/data/production_5M_001_shards/3e/events/{manifest.json,index.json,shards/*.pt}
/cluster/data/production_5M_001_shards/preprocessing_summary.json
```

Both model inputs use `log1p`; every event also retains `ecal_raw_energy` and
`tpad_raw_pe`. Re-submit the same command to reuse completed, matching shards.
Set `EXPECTED_EVENTS_2E=0,EXPECTED_EVENTS_3E=0` only when intentionally building
a dataset whose size is not 5M per class.

## Test scripts belong in ml_ldmx/tests

Reusable checks for behavior that should remain stable between updates belong in
`ml_ldmx/tests`, preferably as `test_*.py` files that can be run together with:

```bash
python -m unittest discover -s tests
```

A script under `scripts/` may still create a small smoke dataset or launch a
cluster preflight job, but regression-style assertions should live in `tests/`.
