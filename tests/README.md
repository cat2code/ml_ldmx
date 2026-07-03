# ml_ldmx/tests

This directory contains reusable regression checks for features that should keep
working as the project changes. Most files are smoke tests: they exercise one
small representative path, assert the important shapes, targets, losses, and
metadata, and then exit without writing training artifacts.

Run the normal test suite from the repository root:

```bash
python -m pytest tests -q
```

The same tests are also compatible with the Python standard library runner:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Useful smoke-test environment variables:

- `ML_LDMX_SMOKE_PROCESSED_DIR`: tiny per-event processed tensor cache for fast local model-family checks.
- `ML_LDMX_PROCESSED_CACHE_ROOT` or `PROCESSED_CACHE_ROOT`: sharded cache root with `2e/events` and `3e/events` subcaches.
- `ML_LDMX_SMOKE_DEVICE`: `cpu`, `cuda`, or `mps` for the maintained model-family smoke test.
- `ML_LDMX_ROOT_DATA`: ROOT data root with `2e/events` and `3e/events`; optional ROOT-backed tests skip when it is unavailable.

`test_model_family_smoke.py` is the main quick regression test. The ROOT-backed
noise and sharded-cache smoke tests are intentionally optional because they
need local or cluster data files.
