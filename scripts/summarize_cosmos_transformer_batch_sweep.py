"""Summarize a COSMOS 2e Transformer batch-size sweep."""

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--slurm-log-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/slurm",
    )
    return parser.parse_args()


def numeric_column(rows, column):
    values = []
    for row in rows:
        token = row.get(column, "").strip().split()
        if not token:
            continue
        try:
            values.append(float(token[0]))
        except ValueError:
            continue
    return values


def gpu_summary(path):
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    utilization = numeric_column(rows, "utilization.gpu [%]")
    memory = numeric_column(rows, "memory.used [MiB]")
    if not utilization or not memory:
        return None
    ordered = sorted(utilization)
    return {
        "average_utilization": sum(utilization) / len(utilization),
        "p95_utilization": ordered[int(0.95 * (len(ordered) - 1))],
        "max_memory_gib": max(memory) / 1024,
    }


def run_summary(campaign_dir, slurm_log_dir, submission):
    run_dir = campaign_dir / submission["run_name"]
    history_path = run_dir / "history.json"
    final_metrics_path = run_dir / "final_metrics.json"
    error_path = slurm_log_dir / f"{submission['job_name']}_{submission['job_id']}.err"
    gpu_path = slurm_log_dir / f"{submission['job_name']}_{submission['job_id']}_gpu.csv"

    history = []
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    epoch = history[-1] if history else {}
    elapsed = epoch.get("train_elapsed_sec")
    train_events = int(0.8 * int(submission["events"]))
    events_per_second = train_events / elapsed if elapsed else None

    if final_metrics_path.is_file():
        status = "complete"
    elif history:
        status = "trained"
    elif error_path.is_file() and "out of memory" in error_path.read_text(
        encoding="utf-8", errors="replace"
    ).lower():
        status = "OOM"
    else:
        status = "no-result"

    return {
        **submission,
        "batch_size": int(submission["batch_size"]),
        "status": status,
        "train_elapsed_sec": elapsed,
        "events_per_second": events_per_second,
        "val_loss": epoch.get("val_loss"),
        "gpu": gpu_summary(gpu_path),
    }


def display(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    args = parse_args()
    campaign_dir = args.campaign_dir.resolve()
    jobs_path = campaign_dir / "submitted_jobs.tsv"
    with jobs_path.open(newline="", encoding="utf-8") as handle:
        submissions = list(csv.DictReader(handle, delimiter="\t"))

    results = [run_summary(campaign_dir, args.slurm_log_dir, row) for row in submissions]
    print("| batch | status | train s | events/s | val loss | GPU avg | GPU p95 | max GiB |")
    print("| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        gpu = result["gpu"] or {}
        print(
            f"| {result['batch_size']} | {result['status']} | "
            f"{display(result['train_elapsed_sec'])} | "
            f"{display(result['events_per_second'])} | "
            f"{display(result['val_loss'], 5)} | "
            f"{display(gpu.get('average_utilization'))}% | "
            f"{display(gpu.get('p95_utilization'))}% | "
            f"{display(gpu.get('max_memory_gib'), 2)} |"
        )

    trained = [result for result in results if result["train_elapsed_sec"] is not None]
    if not trained:
        print("\nNo completed training epoch is available yet.")
        return
    largest = max(trained, key=lambda result: result["batch_size"])
    fastest = max(trained, key=lambda result: result["events_per_second"])
    print(f"\nLargest batch that completed training: {largest['batch_size']}")
    print(
        "Fastest measured training throughput: "
        f"batch {fastest['batch_size']} ({fastest['events_per_second']:.1f} events/s)"
    )


if __name__ == "__main__":
    main()
