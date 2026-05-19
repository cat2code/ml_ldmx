import csv
import json
from pathlib import Path

from mldmx.io.root_files import root_file_sort_key


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_history(history, run_dir):
    save_json(run_dir / "history.json", history)
    csv_path = run_dir / "history.csv"
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row.keys() if not key.endswith("_confusion")})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_config(args, run_dir, data_dir, root_files, event_sources, splits, target_order_counts_by_split):
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload.update(
        {
            "resolved_data_dir": str(data_dir),
            "root_files_used": sorted(
                {source["file"] for source in event_sources},
                key=lambda name: root_file_sort_key(Path(name)),
            ),
            "num_loaded_events": len(event_sources),
            "split_sizes": {key: len(value) for key, value in splits.items()},
            "target_order_counts": target_order_counts_by_split,
        }
    )
    save_json(run_dir / "config.json", payload)
