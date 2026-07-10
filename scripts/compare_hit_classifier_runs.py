"""Compare two saved hit-classifier diagnostic sets event by event."""

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.eval.run_comparison import (
    build_binned_profiles,
    build_multiplicity_profiles,
    comparison_summary,
    flatten_matches,
    interesting_event_groups,
    label_slug,
    load_event_records,
    match_record_sets,
    resolve_diagnostic_path,
)
from ml_ldmx.io.artifacts import save_json
from ml_ldmx.train.logging import setup_logging
from ml_ldmx.viz.run_comparison import (
    plot_accuracy_by_multiplicity,
    plot_binned_profiles,
    plot_paired_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two inspector outputs on matched events and generate report-oriented "
            "difficulty profiles."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run directory, inspection directory, or event-accuracy JSON/CSV; pass exactly twice.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Inspection checkpoint directory to use when PATH is a run directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-partial-match", action="store_true")
    parser.add_argument("--low-accuracy", type=float, default=0.6)
    parser.add_argument("--high-accuracy", type=float, default=0.9)
    parser.add_argument("--difference-threshold", type=float, default=0.1)
    parser.add_argument("--interesting-limit", type=int, default=20)
    return parser.parse_args()


def parse_run_specs(specs):
    if len(specs) != 2:
        raise ValueError("Pass exactly two --run LABEL=PATH arguments.")
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Run specification must be LABEL=PATH, got {spec!r}.")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label or not path.strip():
            raise ValueError(f"Run specification must be LABEL=PATH, got {spec!r}.")
        parsed.append((label, Path(path).expanduser()))
    if parsed[0][0] == parsed[1][0]:
        raise ValueError("Run labels must be distinct.")
    if label_slug(parsed[0][0]) == label_slug(parsed[1][0]):
        raise ValueError("Run labels must have distinct file-safe forms.")
    return parsed


def write_csv(path, rows):
    if not rows:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    def csv_value(value):
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return value

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
    return True


def main():
    args = parse_args()
    if args.num_bins < 2:
        raise ValueError("--num-bins must be at least 2.")
    if args.bootstrap_samples < 0 or args.interesting_limit < 0:
        raise ValueError("--bootstrap-samples and --interesting-limit must be non-negative.")
    if not 0.0 <= args.low_accuracy <= args.high_accuracy <= 1.0:
        raise ValueError("Accuracy thresholds must satisfy 0 <= low <= high <= 1.")
    if args.difference_threshold < 0.0:
        raise ValueError("--difference-threshold must be non-negative.")

    run_specs = parse_run_specs(args.run)
    labels = [label for label, _path in run_specs]
    if args.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "outputs/hit_classifier_comparisons"
            / f"{label_slug(labels[0])}_vs_{label_slug(labels[1])}_{args.split}"
        )
    else:
        output_dir = args.output_dir
    output_dir = output_dir.resolve()
    logger = setup_logging(
        output_dir,
        logger_name="compare_hit_classifier_runs",
        log_filename="comparison.log",
    )

    diagnostic_paths = {}
    record_sets = {}
    for label, path in run_specs:
        diagnostic_path = resolve_diagnostic_path(
            path,
            split=args.split,
            checkpoint=args.checkpoint,
        )
        diagnostic_paths[label] = diagnostic_path
        record_sets[label] = load_event_records(diagnostic_path)
        logger.info("Loaded %s events for %s from %s", len(record_sets[label]), label, diagnostic_path)

    matches, match_info = match_record_sets(
        labels[0],
        record_sets[labels[0]],
        labels[1],
        record_sets[labels[1]],
        allow_partial=args.allow_partial_match,
    )
    rows, slugs = flatten_matches(matches, labels)
    profiles = build_binned_profiles(
        rows,
        labels,
        slugs,
        num_bins=args.num_bins,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    multiplicity_profiles = build_multiplicity_profiles(
        rows,
        labels,
        slugs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary = comparison_summary(
        rows,
        labels,
        slugs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary.update(
        {
            "split": args.split,
            "diagnostic_paths": {
                label: str(path) for label, path in diagnostic_paths.items()
            },
            "event_matching": match_info,
            "profile_bins": args.num_bins,
            "bootstrap_samples": args.bootstrap_samples,
        }
    )
    interesting = interesting_event_groups(
        rows,
        labels,
        slugs,
        low_accuracy=args.low_accuracy,
        high_accuracy=args.high_accuracy,
        difference_threshold=args.difference_threshold,
        limit=args.interesting_limit,
    )

    generated = []
    matched_csv = output_dir / "matched_event_comparison.csv"
    if write_csv(matched_csv, rows):
        generated.append(matched_csv)
    profile_csv = output_dir / "binned_difficulty_profiles.csv"
    if write_csv(profile_csv, profiles):
        generated.append(profile_csv)
    multiplicity_csv = output_dir / "accuracy_by_electron_count.csv"
    if write_csv(multiplicity_csv, multiplicity_profiles):
        generated.append(multiplicity_csv)

    summary_path = output_dir / "comparison_summary.json"
    interesting_path = output_dir / "interesting_events.json"
    save_json(summary_path, summary)
    save_json(interesting_path, interesting)
    generated.extend([summary_path, interesting_path])

    accuracy_profile_path = output_dir / "accuracy_difficulty_profiles.png"
    if plot_binned_profiles(
        profiles,
        labels,
        "accuracy",
        accuracy_profile_path,
        f"{labels[0]} vs {labels[1]}: event accuracy by difficulty",
    ):
        generated.append(accuracy_profile_path)
    energy_profile_path = output_dir / "energy_weighted_accuracy_difficulty_profiles.png"
    if plot_binned_profiles(
        profiles,
        labels,
        "energy_weighted_accuracy",
        energy_profile_path,
        f"{labels[0]} vs {labels[1]}: energy-weighted accuracy by difficulty",
    ):
        generated.append(energy_profile_path)
    paired_path = output_dir / "paired_event_accuracy.png"
    if plot_paired_accuracy(
        rows,
        labels,
        slugs,
        paired_path,
        f"{labels[0]} vs {labels[1]} on matched {args.split} events",
    ):
        generated.append(paired_path)
    multiplicity_path = output_dir / "accuracy_by_electron_count.png"
    if plot_accuracy_by_multiplicity(
        multiplicity_profiles,
        labels,
        multiplicity_path,
        f"{labels[0]} vs {labels[1]} accuracy by electron count",
    ):
        generated.append(multiplicity_path)

    manifest = {
        "labels": labels,
        "split": args.split,
        "matched_events": len(rows),
        "generated_files": [str(path.relative_to(output_dir)) for path in generated],
    }
    save_json(output_dir / "comparison_manifest.json", manifest)
    logger.info("Matched %s events and saved comparison to %s", len(rows), output_dir)


if __name__ == "__main__":
    main()
