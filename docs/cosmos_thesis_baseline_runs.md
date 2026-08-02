# COSMOS thesis baseline night runbook

This launches **eight independent jobs**: two architectures, two source
multiplicities, and two dataset sizes. The phrase `100 000 k` is interpreted
here as **100,000 events**, not 100 million.

The jobs use the context-aware maintained baselines, `ECalTpadGravNet` and
`ECalTpadTransformer`. This is the capacity-oriented choice because it retains
TriggerPadTracks context; it does not assume that the context has already been
proved to improve performance over the ECal-only variants.

## Readiness verdict

The training code is ready after the preflight below succeeds, but do not
submit from a fresh COSMOS clone until the launcher changes in this repository
have been committed and pushed. Also verify that both the `2e` and `3e` caches
contain at least 1,000,000 events. A dataset described as “10 million events”
could otherwise mean 10 million total rather than 10 million per source.

The production jobs are made dependent on one short GPU validation job. If
CUDA, PyTorch Geometric, `torch-cluster`, or the processed cache is broken, the
eight expensive jobs will not start.

This is a capacity-oriented baseline, not a claim that a hyperparameter search
has found the mathematical optimum:

| setting | GravNet | Transformer |
| --- | ---: | ---: |
| hidden dimension | 256 | 256 |
| layers | 6 | 6 |
| learned space / propagated dimensions | 8 / 128 | n/a |
| neighbours `k` | 32 | n/a |
| attention heads / feed-forward dimension | n/a | 8 / 1024 |
| trainable parameters, 2e | 1,000,498 | 4,807,170 |
| trainable parameters, 3e | 1,000,755 | 4,807,427 |

All eight jobs use batch size 8, AdamW with learning rate `3e-4` and weight
decay `1e-4`, dropout 0.1, seed 7, and 20 epochs. Keeping these settings fixed
makes the 100k-versus-1M comparison easier to interpret. `best.pt` retains the
lowest-validation-loss epoch if later epochs do not improve.

COSMOS currently documents `gpua100i` as the shared A100 40 GB batch
partition. The wrapper requests one GPU. The 100k jobs request 48 hours and
the 1M jobs request the documented maximum of 168 hours; a 1M high-capacity
run may take several days rather than finish overnight. See the official
[COSMOS GPU guide](https://lunarc-documentation.readthedocs.io/en/latest/manual/manual_gpu/),
[resource-estimation guide](https://lunarc-documentation.readthedocs.io/en/latest/manual/submitting_jobs/manual_estimating_resources/),
and [job FAQ](https://lunarc-documentation.readthedocs.io/en/latest/manual/faq/manual_faq_jobs/).

## 1. Make the intended Git state available to COSMOS

On the development computer, inspect the standalone `ml_ldmx` repository:

```bash
cd /Users/eliotmontesinopetren/src/mpetren-msceng-ldmx/ml_ldmx
git status --short
git diff --check
git rev-parse HEAD
```

The local worktree currently contains other uncommitted research changes. Make
an intentional commit containing the launcher/runbook changes and any other
changes that should define the training semantics, then push it. A fresh
COSMOS clone only sees pushed commits. Do not accidentally commit generated
datasets or outputs.

## 2. Clone/update and prepare the environment once

Log in and set the two site paths. The repository path below follows the
existing COSMOS project area; change it if the clone or processed cache lives
elsewhere.

```bash
ssh eliotmp@cosmos.lunarc.lu.se

export WORK_ROOT=/projects/hep/fs9/shared/ldmx/users/eliotmp
export REPO_ROOT="$WORK_ROOT/ml_ldmx"
export CACHE_ROOT="$REPO_ROOT/data/processed/production_10M_001_sharded"

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  git clone https://github.com/cat2code/ml_ldmx.git "$REPO_ROOT"
fi

cd "$REPO_ROOT"
git pull --ff-only
git status --short
git rev-parse HEAD

module --force purge
module load GCCcore/13.2.0
module load Python/3.11.5

if [[ ! -f .venv/bin/activate ]]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

export VENV_DIR="$REPO_ROOT/.venv"
mkdir -p outputs/slurm
```

If `Python/3.11.5` is no longer an available short module name, inspect the
current name with `module spider Python/3.11.5`; LUNARC also documents the
fully qualified name `Python/3.11.5-GCCcore-13.2.0`.

If more than one allocation is listed by `projinfo`, add the correct
`--account=<project>` in the submission block below. Do not guess an account.

```bash
projinfo
sinfo -p gpua100i -o '%P %l %a %D'
```

## 3. Validate the processed cache before using GPU time

This checks index completeness, listed shard existence, sample tensor
contents, transforms, and the minimum event count independently for `2e` and
`3e`:

```bash
cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

python - "$CACHE_ROOT" <<'PY'
import sys
from pathlib import Path

from ml_ldmx.datasets.ecal_tpad_shards import validate_ml_ready_sharded_cache

root = Path(sys.argv[1]).resolve()
for label in ("2e", "3e"):
    cache = root / label / "events"
    manifest, index = validate_ml_ready_sharded_cache(cache)
    spec = manifest["cache_spec"]
    count = int(index["num_events"])
    assert count >= 1_000_000, f"{label} has only {count:,} events"
    assert spec.get("ecal_energy_transform") == "log1p", (label, spec)
    assert spec.get("tpad_pe_transform") == "log1p", (label, spec)
    print(
        f"{label}: {count:,} events, {len(index['shards'])} shards, "
        f"target rule={spec.get('hard_origin_target_rule', 'legacy/unspecified')}"
    )
PY

python -m unittest discover -s tests -p 'test_*.py'
```

Do not continue if either count is below 1,000,000 or either transform is not
`log1p`. The training transform must match what was stored in the cache.

## 4. Submit the GPU gate and all eight jobs

Copy this entire block from the `ml_ldmx` repository root. Slurm will queue
the jobs; submitting eight jobs does not require eight GPUs to be free now.

```bash
cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"
mkdir -p outputs/slurm
set -euo pipefail
unset SOURCE_LABEL ELECTRON_COUNT RESUME

export CAMPAIGN="thesis_baselines_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_ROOT="$REPO_ROOT/outputs/cosmos_baselines/$CAMPAIGN"
mkdir -p "$OUTPUT_ROOT"

# If projinfo shows more than one project, add --account=YOUR_PROJECT after
# --parsable in both sbatch commands below.

PREFLIGHT_SUBMISSION=$(sbatch --parsable \
  --time=00:30:00 \
  --export="ALL,REPO_ROOT=$REPO_ROOT,VENV_DIR=$VENV_DIR,PROCESSED_CACHE_ROOT=$CACHE_ROOT,EVENTS_PER_SOURCE=10,MIN_EVENTS_PER_SOURCE=1000000,ECAL_ENERGY_TRANSFORM=log1p,TPAD_PE_TRANSFORM=log1p,HIDDEN_DIM=256,NUM_LAYERS=6,NUM_HEADS=8,DIM_FEEDFORWARD=1024,SPACE_DIMENSIONS=8,PROPAGATE_DIMENSIONS=128,K=32" \
  tests/cosmos_validate_gpu.sbatch)
PREFLIGHT_JOB_ID=${PREFLIGHT_SUBMISSION%%;*}
printf '%s\n' "$PREFLIGHT_JOB_ID" | tee "$OUTPUT_ROOT/preflight_job_id.txt"

JOB_IDS_FILE="$OUTPUT_ROOT/submitted_jobs.tsv"
printf 'job_id\tjob_name\trun_name\n' > "$JOB_IDS_FILE"

submit_run() {
  local job_name=$1
  local model=$2
  local source_label=$3
  local electron_count=$4
  local event_count=$5
  local walltime=$6
  local run_name=$7
  local submission job_id

  submission=$(sbatch --parsable \
    --dependency="afterok:$PREFLIGHT_JOB_ID" \
    --job-name="$job_name" \
    --time="$walltime" \
    --mem=64G \
    --export="ALL,REPO_ROOT=$REPO_ROOT,VENV_DIR=$VENV_DIR,PROCESSED_CACHE_ROOT=$CACHE_ROOT,OUTPUT_ROOT=$OUTPUT_ROOT,MODEL=$model,SOURCE_LABEL=$source_label,ELECTRON_COUNT=$electron_count,EVENTS_PER_SOURCE=$event_count,EPOCHS=20,BATCH_SIZE=8,LR=3e-4,WEIGHT_DECAY=1e-4,HIDDEN_DIM=256,NUM_LAYERS=6,NUM_HEADS=8,DIM_FEEDFORWARD=1024,DROPOUT=0.1,SPACE_DIMENSIONS=8,PROPAGATE_DIMENSIONS=128,K=32,GRAD_CLIP=1.0,SHARD_CACHE_SIZE=1,CACHE_MODEL_VIEWS=1,SEED=7,ECAL_ENERGY_TRANSFORM=log1p,TPAD_PE_TRANSFORM=log1p,RUN_NAME=$run_name" \
    scripts/sbatch/cosmos_train_baseline.sbatch)
  job_id=${submission%%;*}
  printf '%s\t%s\t%s\n' "$job_id" "$job_name" "$run_name" | tee -a "$JOB_IDS_FILE"
}

submit_run ml_g2_100k ECalTpadGravNet     2e 2 100000  2-00:00:00 tpad_gravnet_2e_100k_h256_l6_seed7
submit_run ml_g3_100k ECalTpadGravNet     3e 3 100000  2-00:00:00 tpad_gravnet_3e_100k_h256_l6_seed7
submit_run ml_g2_1m   ECalTpadGravNet     2e 2 1000000 7-00:00:00 tpad_gravnet_2e_1m_h256_l6_seed7
submit_run ml_g3_1m   ECalTpadGravNet     3e 3 1000000 7-00:00:00 tpad_gravnet_3e_1m_h256_l6_seed7

submit_run ml_t2_100k ECalTpadTransformer 2e 2 100000  2-00:00:00 tpad_transformer_2e_100k_h256_l6_seed7
submit_run ml_t3_100k ECalTpadTransformer 3e 3 100000  2-00:00:00 tpad_transformer_3e_100k_h256_l6_seed7
submit_run ml_t2_1m   ECalTpadTransformer 2e 2 1000000 7-00:00:00 tpad_transformer_2e_1m_h256_l6_seed7
submit_run ml_t3_1m   ECalTpadTransformer 3e 3 1000000 7-00:00:00 tpad_transformer_3e_1m_h256_l6_seed7

echo "campaign: $CAMPAIGN"
echo "output root: $OUTPUT_ROOT"
cat "$JOB_IDS_FILE"
```

Record the printed output root. Do not pull, switch commits, or edit the shared
clone while these jobs are queued or running: every job must execute the same
code. On a later SSH login, restore the concrete paths before monitoring:

```bash
export REPO_ROOT=/projects/hep/fs9/shared/ldmx/users/eliotmp/ml_ldmx
export OUTPUT_ROOT=/paste/the/printed/campaign/output/root
cd "$REPO_ROOT"
```

Important details handled by the wrapper:

- `2e` gets the true two-class label space `1 2`; `3e` gets `1 2 3`.
- Every run name and Slurm job name is unique.
- The Git commit, dirty status, exact Python command, model parameter count,
  host, CUDA environment, start/end time, exit status, and `/usr/bin/time -v`
  statistics are logged on normal process exit. `/usr/bin/time -v` writes its
  statistics to the Slurm `.err` file even when training succeeds.
- GPU utilization, memory, and power are sampled once per minute into
  `outputs/slurm/<job-name>_<job-id>_gpu.csv`.
- The package is installed once before submission. The eight jobs only
  activate and import-check the shared environment, so they do not race while
  modifying one `.venv`.
- Checkpoints are saved after each completed epoch. `--no-requeue` avoids a
  node-failure restart silently writing a fresh run into the same directory.

## 5. Monitor the campaign

```bash
PREFLIGHT_JOB_ID=$(<"$OUTPUT_ROOT/preflight_job_id.txt")
PRODUCTION_JOB_IDS=$(tail -n +2 "$OUTPUT_ROOT/submitted_jobs.tsv" | cut -f1 | paste -sd, -)

squeue -j "$PREFLIGHT_JOB_ID,$PRODUCTION_JOB_IDS"
jobinfo -u "$USER"

# Replace the ID with one from submitted_jobs.tsv.
squeue --start -j JOB_ID
tail -F outputs/slurm/ml_t3_1m_JOB_ID.out
```

If the GPU gate fails, the production jobs remain blocked by their dependency.
Inspect `outputs/slurm/ml_ldmx_validate_gpu_<preflight-id>.{out,err}`, fix the
problem, and cancel the blocked jobs before resubmitting:

```bash
tail -n +2 "$OUTPUT_ROOT/submitted_jobs.tsv" | cut -f1 | while read -r job_id; do
  [[ -n "$job_id" ]] && scancel --quiet "$job_id" || true
done
```

## 6. Record resource use after completion

```bash
PREFLIGHT_JOB_ID=$(<"$OUTPUT_ROOT/preflight_job_id.txt")
PRODUCTION_JOB_IDS=$(tail -n +2 "$OUTPUT_ROOT/submitted_jobs.tsv" | cut -f1 | paste -sd, -)

sacct -j "$PREFLIGHT_JOB_ID,$PRODUCTION_JOB_IDS" \
  --units=G \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,Timelimit,TotalCPU,ReqMem,MaxRSS,AllocCPUS,AllocTRES%40 \
  | tee "$OUTPUT_ROOT/slurm_accounting.txt"

ls -lh outputs/slurm/*_gpu.csv
```

For each successful run, require all of the following before calling it
complete:

```text
config.json
run_overview.json
model_architecture.txt
history.csv
train.log
checkpoints/best.pt
checkpoints/latest.pt
final_metrics.json
test_hit_origin_confusion_matrix.png
```

If a job reaches its walltime, the active epoch is lost but the last completed
epoch remains in `checkpoints/latest.pt`, including the best validation loss
known at that point. Resume with exactly the same source, event count, seed,
architecture, optimizer settings, output root, and run name; set `RESUME` to
that `latest.pt`, and keep `EPOCHS=20` because it is the total target epoch
count rather than the number of additional epochs. Accounting and the
one-minute GPU monitor are best-effort if Slurm sends a hard timeout; `sacct`
is the authoritative post-run resource record.

## 7. Thesis-use caveats for later plotting

Use `checkpoints/best.pt` for the later thesis analysis. The top-level final
metrics describe the final in-memory epoch, which need not be the
validation-selected best epoch.

The 100k and 1M loaders select the first 100k and first 1M cached events and
then create a deterministic split. Their automatically generated test sets are
not one common external held-out set. The runs are still valid training runs,
but a thesis scaling comparison should later evaluate all relevant best
checkpoints on the same held-out evaluation cache, or state clearly that the
reported test samples differ.
