"""Generate report-facing validation plots from saved hit-classifier checkpoints."""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import inspect_hit_classifier_run as inspection

from ml_ldmx.eval.hit_classifier_baseline import iter_event_predictions
from ml_ldmx.eval.permutation_aligned import (
    accuracy_coverage_profiles,
    aligned_event_metrics,
    aligned_event_prediction,
    assign_global_layers,
    binned_event_profile,
    calibration_bins,
    confusion_counts,
    event_layer_metrics,
    global_layer_z,
    layer_profiles,
)
from ml_ldmx.io.artifacts import save_json
from ml_ldmx.train.checkpoints import checkpoint_hard_origin_target_rule
from ml_ldmx.train.logging import setup_logging
from ml_ldmx.train.utils import resolve_device
from ml_ldmx.viz.permutation_aligned import (
    DEPTH_RANGES,
    plot_accuracy_coverage,
    plot_all_confusions,
    plot_confidence_by_layer,
    plot_entropy_density,
    plot_event_accuracy_distributions,
    plot_event_accuracy_distributions_with_boxplots,
    plot_event_confidence_density,
    plot_layer_accuracy_distribution,
    plot_separation_density,
)


DEFAULT_RUNS = (
    (
        "2e",
        PROJECT_ROOT
        / "outputs/summed_origin_energy_20k/transformer_2e_20k_seed7_summed_origin",
    ),
    (
        "3e",
        PROJECT_ROOT
        / "outputs/summed_origin_energy_20k/transformer_3e_20k_seed7_summed_origin",
    ),
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/plots_development/003_refined_validation_20k"
)
PROFILE_KEYS = (
    "energy_weighted_min_centroid_distance_moliere",
    "first_3_layers_energy_weighted_min_centroid_distance_moliere",
    "energy_weighted_min_width_normalized_separation",
    "first_3_layers_energy_weighted_min_width_normalized_separation",
    "mean_confidence",
    "p10_confidence",
    "min_confidence",
    "confidence_standard_deviation",
    "fraction_confidence_below_0p8",
    "mean_normalized_entropy",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one inference pass over saved validation splits and generate "
            "permutation-aligned report plots."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_DIR"),
        help="Saved run to analyze. Repeat for multiple runs. Defaults to 2e and 3e.",
    )
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--moliere-radius-mm", type=float, default=25.0)
    parser.add_argument("--expected-ecal-layers", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def _inspection_cli(args):
    return SimpleNamespace(
        processed_dir=None,
        processed_cache=None,
        processed_cache_root=None,
        processed_source=None,
        data_root=None,
        events_per_source=None,
        shard_cache_size=None,
        batch_size=args.batch_size,
        event_diagnostic_radius_mm=None,
        num_events=0,
        evaluation_hard_origin_target_rule=None,
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(row.get(key)) for key in fieldnames})
    return path


def _confusion_rows(matrix, layer_start=None, layer_stop=None):
    matrix = np.asarray(matrix, dtype=np.int64)
    row_sums = matrix.sum(axis=1)
    rows = []
    for true_class in range(matrix.shape[0]):
        for predicted_class in range(matrix.shape[1]):
            count = int(matrix[true_class, predicted_class])
            rows.append(
                {
                    "layer_start": layer_start,
                    "layer_stop": layer_stop,
                    "true_group": true_class + 1,
                    "aligned_predicted_group": predicted_class + 1,
                    "hit_count": count,
                    "row_fraction": (
                        None
                        if row_sums[true_class] == 0
                        else count / int(row_sums[true_class])
                    ),
                }
            )
    return rows


def _save_hit_cache(path, events):
    event_idx = np.concatenate(
        [
            np.full(event.true_class.shape, event.event_idx, dtype=np.int32)
            for event in events
        ]
    )
    np.savez_compressed(
        path,
        event_idx=event_idx,
        layer=np.concatenate([event.layer for event in events]).astype(np.uint8),
        true_group=np.concatenate([event.true_class for event in events]).astype(np.uint8)
        + 1,
        predicted_group=np.concatenate(
            [event.predicted_class for event in events]
        ).astype(np.uint8)
        + 1,
        aligned_predicted_group=np.concatenate(
            [event.aligned_predicted_class for event in events]
        ).astype(np.uint8)
        + 1,
        confidence=np.concatenate([event.confidence for event in events]).astype(
            np.float32
        ),
        raw_reconstructed_energy=np.concatenate(
            [event.raw_energy for event in events]
        ).astype(np.float32),
    )
    return path


def _summary(events, records, confusion, ece):
    num_events = len(events)
    num_hits = int(sum(event.true_class.size for event in events))
    aligned_correct = int(sum(event.aligned_correct.sum() for event in events))
    ordinary_correct = int(sum(event.ordinary_correct.sum() for event in events))
    total_energy = float(sum(np.clip(event.raw_energy, 0.0, None).sum() for event in events))
    aligned_correct_energy = float(
        sum(
            np.clip(event.raw_energy, 0.0, None)[event.aligned_correct].sum()
            for event in events
        )
    )
    summary = {
        "num_events": num_events,
        "num_hits": num_hits,
        "ordinary_pooled_hit_accuracy": ordinary_correct / num_hits,
        "ordinary_macro_event_accuracy": statistics.fmean(
            record["ordinary_event_accuracy"] for record in records
        ),
        "pooled_aligned_hit_accuracy": aligned_correct / num_hits,
        "macro_aligned_event_accuracy": statistics.fmean(
            record["aligned_event_accuracy"] for record in records
        ),
        "macro_aligned_event_accuracy_median": statistics.median(
            record["aligned_event_accuracy"] for record in records
        ),
        "aligned_energy_weighted_accuracy_under_hit_optimal_mapping": (
            None if total_energy <= 0.0 else aligned_correct_energy / total_energy
        ),
        "macro_aligned_event_energy_weighted_accuracy": statistics.fmean(
            record["aligned_energy_weighted_accuracy"] for record in records
        ),
        "nonidentity_mapping_events": sum(
            not record["mapping_is_identity"] for record in records
        ),
        "nonidentity_mapping_event_fraction": statistics.fmean(
            not record["mapping_is_identity"] for record in records
        ),
        "expected_calibration_error": float(ece),
    }
    if int(np.trace(confusion)) != aligned_correct:
        raise RuntimeError("Confusion diagonal does not reproduce aligned correct-hit count.")
    return summary


def _validate_analysis(events, analysis, expected_layers):
    overall = analysis["confusion"]
    depth_sum = np.zeros_like(overall)
    for matrix in analysis["depth_confusions"].values():
        depth_sum += matrix
    if not np.array_equal(overall, depth_sum):
        raise RuntimeError("The ECal depth-range confusion matrices do not sum to the total.")
    if len(analysis["layer_profiles"]) != int(expected_layers):
        raise RuntimeError("Layer profiles do not cover every expected ECal layer.")
    if sum(row["num_hits"] for row in analysis["calibration_rows"]) != analysis["summary"]["num_hits"]:
        raise RuntimeError("Calibration bins do not cover every retained hit.")
    final_hit = analysis["hit_coverage_rows"][-1]
    final_event = analysis["event_coverage_rows"][-1]
    if not np.isclose(
        final_hit["aligned_hit_accuracy"],
        analysis["summary"]["pooled_aligned_hit_accuracy"],
    ):
        raise RuntimeError("Full hit coverage does not reproduce pooled aligned accuracy.")
    if not np.isclose(
        final_event["macro_aligned_event_accuracy"],
        analysis["summary"]["macro_aligned_event_accuracy"],
    ):
        raise RuntimeError("Full event coverage does not reproduce macro aligned accuracy.")
    if any(event.layer is None for event in events):
        raise RuntimeError("At least one event is missing global layer assignments.")


def analyze_run(label, run_dir, command_args, device, logger):
    run_dir = Path(run_dir).resolve()
    checkpoint_path = inspection.resolve_checkpoint_path(
        run_dir,
        Path(command_args.checkpoint),
    )
    checkpoint = inspection._load_checkpoint(checkpoint_path)
    config = inspection._load_json(run_dir / "config.json")
    train_args = inspection._training_args(
        checkpoint,
        config,
        _inspection_cli(command_args),
    )
    if command_args.batch_size is not None:
        train_args.batch_size = int(command_args.batch_size)

    logger.info("%s: loading saved run %s", label, run_dir)
    events, _event_sources, data_dir, _root_files = inspection.training.load_events(
        train_args,
        logger,
    )
    indices = inspection.validate_saved_split(
        events,
        checkpoint,
        config,
        command_args.split,
    )
    events = inspection.restore_event_preprocessing(events, checkpoint, train_args)
    model, view_fn = inspection.restore_model(checkpoint, train_args, device)
    aligned_events = []
    for completed, prediction in enumerate(
        iter_event_predictions(
            model,
            events,
            indices,
            view_fn,
            train_args,
            device,
        ),
        start=1,
    ):
        aligned_events.append(
            aligned_event_prediction(
                prediction,
                num_classes=len(train_args.valid_labels),
            )
        )
        if command_args.progress_every > 0 and completed % command_args.progress_every == 0:
            logger.info("%s: inferred %s/%s validation events", label, completed, len(indices))
    aligned_events.sort(key=lambda event: event.split_position)
    if len(aligned_events) != len(indices):
        raise RuntimeError(
            f"{label}: inferred {len(aligned_events)} events for a {len(indices)}-event split."
        )

    layer_z = global_layer_z(
        aligned_events,
        expected_layers=command_args.expected_ecal_layers,
    )
    assign_global_layers(aligned_events, layer_z)
    records = [
        aligned_event_metrics(
            event,
            moliere_radius_mm=command_args.moliere_radius_mm,
            early_layers=3,
        )
        for event in aligned_events
    ]
    layer_rows = [
        row
        for event in aligned_events
        for row in event_layer_metrics(event)
    ]
    layer_profile_rows = layer_profiles(
        layer_rows,
        bootstrap_samples=command_args.bootstrap_samples,
        seed=command_args.bootstrap_seed,
    )
    overall_confusion = confusion_counts(
        aligned_events,
        num_classes=len(train_args.valid_labels),
    )
    depth_confusions = {
        f"{start:02d}_{stop:02d}": confusion_counts(
            aligned_events,
            num_classes=len(train_args.valid_labels),
            layer_min=start,
            layer_max=stop,
        )
        for start, stop in DEPTH_RANGES
    }
    calibration_rows, ece = calibration_bins(aligned_events, num_bins=10)
    hit_coverage_rows, event_coverage_rows = accuracy_coverage_profiles(
        aligned_events,
        points=20,
    )
    binned_profiles = {
        key: binned_event_profile(
            records,
            x_key=key,
            bootstrap_samples=command_args.bootstrap_samples,
            seed=command_args.bootstrap_seed,
        )
        for key in PROFILE_KEYS
    }
    summary = _summary(aligned_events, records, overall_confusion, ece)
    analysis = {
        "events": aligned_events,
        "event_records": records,
        "layer_rows": layer_rows,
        "layer_profiles": layer_profile_rows,
        "confusion": overall_confusion,
        "depth_confusions": depth_confusions,
        "calibration_rows": calibration_rows,
        "ece": ece,
        "hit_coverage_rows": hit_coverage_rows,
        "event_coverage_rows": event_coverage_rows,
        "binned_profiles": binned_profiles,
        "summary": summary,
        "layer_z_mm": layer_z.tolist(),
        "run_metadata": {
            "label": label,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch_zero_based": int(checkpoint.get("epoch", -1)),
            "checkpoint_epoch_human": int(checkpoint.get("epoch", -1)) + 1,
            "model": train_args.model,
            "valid_labels": list(train_args.valid_labels),
            "saved_split": command_args.split,
            "saved_split_events": len(indices),
            "saved_split_lengths": {
                split: len(values)
                for split, values in checkpoint["splits"].items()
            },
            "data_dir": (
                [str(path) for path in data_dir]
                if isinstance(data_dir, (list, tuple))
                else str(data_dir)
            ),
            "hard_origin_target_rule": checkpoint_hard_origin_target_rule(checkpoint),
            "target_mode": train_args.target_mode,
        },
    }
    _validate_analysis(aligned_events, analysis, command_args.expected_ecal_layers)
    logger.info(
        "%s: macro aligned accuracy %.4f, pooled aligned accuracy %.4f",
        label,
        summary["macro_aligned_event_accuracy"],
        summary["pooled_aligned_hit_accuracy"],
    )
    return analysis


def save_analysis_data(output_dir, label, analysis):
    data_dir = Path(output_dir) / "data" / label
    data_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    generated.append(save_csv(data_dir / "event_metrics.csv", analysis["event_records"]))
    save_json(data_dir / "event_metrics.json", analysis["event_records"])
    generated.append(data_dir / "event_metrics.json")
    generated.append(save_csv(data_dir / "event_layer_metrics.csv", analysis["layer_rows"]))
    generated.append(save_csv(data_dir / "layer_profiles.csv", analysis["layer_profiles"]))
    generated.append(
        save_csv(
            data_dir / "overall_confusion.csv",
            _confusion_rows(analysis["confusion"]),
        )
    )
    depth_rows = []
    for start, stop in DEPTH_RANGES:
        depth_rows.extend(
            _confusion_rows(
                analysis["depth_confusions"][f"{start:02d}_{stop:02d}"],
                layer_start=start,
                layer_stop=stop,
            )
        )
    generated.append(save_csv(data_dir / "depth_confusions.csv", depth_rows))
    generated.append(save_csv(data_dir / "calibration.csv", analysis["calibration_rows"]))
    generated.append(save_csv(data_dir / "hit_accuracy_coverage.csv", analysis["hit_coverage_rows"]))
    generated.append(
        save_csv(data_dir / "event_accuracy_coverage.csv", analysis["event_coverage_rows"])
    )
    binned_rows = [
        row
        for profile in analysis["binned_profiles"].values()
        for row in profile
    ]
    generated.append(save_csv(data_dir / "binned_event_profiles.csv", binned_rows))
    generated.append(_save_hit_cache(data_dir / "aligned_hit_cache.npz", analysis["events"]))
    return generated


def make_plots(output_dir, analyses):
    output_dir = Path(output_dir)
    generated = []
    generated.extend(
        plot_event_accuracy_distributions(
            analyses,
            output_dir / "01_event_accuracy_distribution",
        )
    )
    generated.extend(
        plot_event_accuracy_distributions_with_boxplots(
            analyses,
            output_dir / "01b_event_accuracy_distribution_boxplot_experiment",
        )
    )
    generated.extend(
        plot_all_confusions(
            analyses,
            output_dir / "02_combined_hit_confusion_matrices",
        )
    )
    for plot_number, (label, analysis) in enumerate(analyses.items(), start=4):
        generated.extend(
            plot_layer_accuracy_distribution(
                analysis,
                label,
                output_dir
                / f"{plot_number:02d}_{label}_event_layer_hit_accuracy_density",
                metric="hit",
            )
        )
    for plot_number, (label, analysis) in enumerate(analyses.items(), start=6):
        generated.extend(
            plot_layer_accuracy_distribution(
                analysis,
                label,
                output_dir
                / f"{plot_number:02d}_{label}_event_layer_energy_weighted_accuracy_density",
                metric="energy",
            )
        )
    generated.extend(
        plot_separation_density(
            analyses,
            output_dir / "08_energy_weighted_separation_in_moliere_units",
            metric_keys=(
                ("energy_weighted_min_centroid_distance_moliere", "all ECal layers"),
            ),
            xlabel=r"minimum energy-weighted centroid distance $d_{\min}/R_{\mathrm{M}}$",
            figure_title=(
                "Event accuracy versus shower separation "
                r"($R_{\mathrm{M}}=25$ mm)"
            ),
            x_reference=1.0,
            x_range=(0.0, 4.0),
            clip_to_range=True,
        )
    )
    generated.extend(
        plot_separation_density(
            analyses,
            output_dir / "08b_energy_weighted_separation_log_x_experiment",
            metric_keys=(
                ("energy_weighted_min_centroid_distance_moliere", "all ECal layers"),
            ),
            xlabel=(
                r"minimum energy-weighted centroid distance $d_{\min}/R_{\mathrm{M}}$ "
                "(log scale)"
            ),
            figure_title=(
                "Event accuracy versus shower separation on a logarithmic scale "
                r"($R_{\mathrm{M}}=25$ mm)"
            ),
            x_reference=1.0,
            x_range=(0.005, 4.0),
            x_scale="log",
            clip_to_range=True,
        )
    )
    generated.extend(
        plot_separation_density(
            analyses,
            output_dir / "09_energy_weighted_width_normalized_separation",
            metric_keys=(
                (
                    "energy_weighted_min_width_normalized_separation",
                    "all ECal layers",
                ),
                (
                    "first_3_layers_energy_weighted_min_width_normalized_separation",
                    "ECal layers 1–3",
                ),
            ),
            xlabel=(
                "minimum centroid distance / combined shower width "
                "(log scale)"
            ),
            figure_title=(
                "Event accuracy versus energy-weighted "
                "width-normalized separation"
            ),
            x_scale="log",
        )
    )
    generated.extend(
        plot_event_confidence_density(
            analyses,
            output_dir / "10_event_confidence_vs_event_accuracy",
        )
    )
    generated.extend(
        plot_entropy_density(
            analyses,
            output_dir / "11_mean_normalized_entropy_vs_event_accuracy",
        )
    )
    generated.extend(
        plot_accuracy_coverage(
            analyses,
            output_dir / "13_accuracy_coverage",
        )
    )
    generated.extend(
        plot_confidence_by_layer(
            analyses,
            output_dir / "14_confidence_and_accuracy_by_ecal_layer",
        )
    )
    return generated


def write_readme(output_dir, analyses):
    output_dir = Path(output_dir)
    summary_lines = [
        "| Sample | Events | Hits | Mean event accuracy | Pooled hit accuracy | Energy-weighted hit accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, analysis in analyses.items():
        summary = analysis["summary"]
        summary_lines.append(
            "| {label} | {events:,} | {hits:,} | {macro:.4f} | {pooled:.4f} | {energy:.4f} |".format(
                label=label,
                events=summary["num_events"],
                hits=summary["num_hits"],
                macro=summary["macro_aligned_event_accuracy"],
                pooled=summary["pooled_aligned_hit_accuracy"],
                energy=summary[
                    "aligned_energy_weighted_accuracy_under_hit_optimal_mapping"
                ],
            )
        )
    text = """# Refined 20k transformer validation plots

This bundle uses the saved validation split and best checkpoint from each 20k
transformer run. Each saved validation split contains 3,000 events from the
earlier 80/15/5 split. No test events are used.

The numerical electron-group labels have no physical ordering. For each event,
one permutation is selected to maximize the number of correctly grouped ECal
hits. That same whole-event permutation is then fixed for every depth,
confidence, energy, and confusion diagnostic. No layer or confidence subset is
matched again. Plot labels use `accuracy` without repeating this convention.

The reported mean event accuracy first calculates the hit accuracy within each
event and then gives every event equal weight. Confusion matrices follow the
standard convention and pool hits before row normalization.

## Summary

{summary}

## Plot index

1. `01_event_accuracy_distribution` overlays the 2e and 3e event-accuracy
   histograms and reports their medians. The `01b` variant experimentally adds
   a narrow shared-axis box-plot summary below the histograms.
2. `02_combined_hit_confusion_matrices` contains the overall 2e and 3e
   confusion matrices in the first row. The next two rows compare layers 1–20
   and 21–32 for each sample. The saved detector geometry contains 32 ECal
   layers.
4. `04_2e_event_layer_hit_accuracy_density` and
   `05_3e_event_layer_hit_accuracy_density` show event-layer hit accuracy with
   rectangular count bins.
5. `06_2e_event_layer_energy_weighted_accuracy_density` and
   `07_3e_event_layer_energy_weighted_accuracy_density` show the corresponding
   energy-weighted accuracy.
8. `08_energy_weighted_separation_in_moliere_units` uses reconstructed hit
   energy times the simulated origin fraction over all ECal layers. The red
   line marks one Molière radius, using 25 mm as an interpretive scale rather
   than a hard resolution limit. The `08b` variant uses a logarithmic
   horizontal axis.
9. `09_energy_weighted_width_normalized_separation` normalizes centroid distance
   by the combined RMS shower width and uses a logarithmic horizontal axis.
10. `10_event_confidence_vs_event_accuracy` tests mean event
    confidence using rectangular density bins.
11. `11_mean_normalized_entropy_vs_event_accuracy` retains the entropy
    diagnostic from the previous multi-metric figure.
12. Plot 12 from the previous iteration is intentionally omitted.
13. `13_accuracy_coverage` combines the 2e and 3e curves into one hit-selection
    panel and one event-selection panel.
14. `14_confidence_and_accuracy_by_ecal_layer` compares depth-dependent
    confidence with depth-dependent accuracy.

Every rectangular density plot colors zero-count bins at the lower end of the
count scale. White points and vertical bars show equal-population event-bin
means with 95% event-bootstrap confidence intervals.

Each plot is saved as a PNG for quick review and as a PDF for later use in
Overleaf. The `data` directory contains the plot tables and a compact aligned
hit cache. `summary.json` and `manifest.json` record the definitions and source
checkpoints.

These plots are development results because the best checkpoints were selected
using the same validation events. Final headline performance should be reported
on a held-out test set after the plot design is fixed.
""".replace("{summary}", "\n".join(summary_lines))
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    args = parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative.")
    if args.expected_ecal_layers <= 0:
        raise ValueError("--expected-ecal-layers must be positive.")
    if args.moliere_radius_mm <= 0:
        raise ValueError("--moliere-radius-mm must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        output_dir,
        logger_name="plot_permutation_aligned_validation",
        log_filename="analysis.log",
    )
    device = resolve_device(args.device, logger)
    run_specs = (
        [(label, Path(path)) for label, path in args.run]
        if args.run
        else list(DEFAULT_RUNS)
    )
    labels = [label for label, _path in run_specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Run labels must be unique.")

    analyses = {}
    for label, run_dir in run_specs:
        analyses[label] = analyze_run(label, run_dir, args, device, logger)

    generated = []
    for label, analysis in analyses.items():
        generated.extend(save_analysis_data(output_dir, label, analysis))
    generated.extend(make_plots(output_dir, analyses))

    summary = {label: analysis["summary"] for label, analysis in analyses.items()}
    summary_path = output_dir / "summary.json"
    save_json(summary_path, summary)
    generated.append(summary_path)
    readme_path = write_readme(output_dir, analyses)
    generated.append(readme_path)

    manifest = {
        "analysis": "permutation-aligned hit-grouping validation",
        "split": args.split,
        "device": str(device),
        "moliere_radius_mm": float(args.moliere_radius_mm),
        "expected_ecal_layers": int(args.expected_ecal_layers),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "alignment_scope": "one hit-count-optimal permutation per complete event",
        "alignment_reused_for_all_subsets": True,
        "confusion_weighting": "pooled hit counts, row normalized for display",
        "reported_average_accuracy": "mean of per-event hit accuracies",
        "energy_mapping": "hit-count-optimal mapping, not separately energy optimized",
        "separation_weighting": (
            "raw reconstructed ECal hit energy multiplied by simulated "
            "per-origin deposited-energy fraction"
        ),
        "runs": {
            label: analysis["run_metadata"]
            for label, analysis in analyses.items()
        },
        "layer_z_mm": {
            label: analysis["layer_z_mm"]
            for label, analysis in analyses.items()
        },
        "generated_files": sorted(
            str(Path(path).resolve().relative_to(output_dir))
            for path in generated
        ),
    }
    manifest_path = output_dir / "manifest.json"
    save_json(manifest_path, manifest)
    logger.info("Saved %s generated artifacts to %s", len(generated) + 1, output_dir)


if __name__ == "__main__":
    main()
