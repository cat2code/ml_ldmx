# ml_ldmx/scripts

This directory contains runnable Python and sbatch entry points for tasks the
user wants to perform: preprocessing, training, benchmarking, environment
checks, and cluster launch jobs.

Scripts should call reusable functionality from `ml_ldmx/src/ml_ldmx`; they
should not become the place where maintained features are implemented.

## Test scripts belong in ml_ldmx/tests

Reusable checks for behavior that should remain stable between updates belong in
`ml_ldmx/tests`, preferably as `test_*.py` files that can be run together with:

```bash
python -m pytest tests -q
```

A script under `scripts/` may still create a small smoke dataset or launch a
cluster preflight job, but regression-style assertions should live in `tests/`.
