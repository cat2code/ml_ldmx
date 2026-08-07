#!/usr/bin/env python3
"""Evaluate a trained contributor-set slot model with and without TPad tokens."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_ecal_tpad_contributor_set_slot_model as training  # noqa: E402

from ml_ldmx.datasets.ecal_tpad_loading import (  # noqa: E402
    apply_variable_count_target_mode,
    filter_noise_tensor_event,
)
from ml_ldmx.datasets.preprocess import normalize_event_continuous_features  # noqa: E402
from ml_ldmx.eval.ecal_tpad_contributor_set_slot_model import evaluate  # noqa: E402
from ml_ldmx.io.artifacts import save_json  # noqa: E402
from ml_ldmx.models import ECalTpadContributorSetSlotModel  # noqa: E402
from ml_ldmx.train.checkpoints import (  # noqa: E402
    checkpoint_hard_origin_target_rule,
    read_checkpoint,
)
from ml_ldmx.train.logging import setup_logging  # noqa: E402
from ml_ldmx.train.utils import resolve_device  # noqa: E402
from ml_ldmx.viz.contributor_set_ablation import (  # noqa: E402
    plot_confusion_ablation,
    plot_count_ablation,
    plot_fraction_mae_ablation,
    plot_task_metric_ablation,
)
from ml_ldmx.viz.contributor_sets import (  # noqa: E402
    contributor_set_labels,
    plot_mixed_probability_diagnostics,
)


PATH_ARGUMENTS = (
    "data_root",
    "processed_dir",
    "processed_cache",
    "processed_cache_root",
    "output_root",
    "output_dir",
    "resume",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run paired TPad-token ablation on a saved "
            "ECalTpadContributorSetSlotModel checkpoint."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-plot-hits", type=int, default=100_000)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--processed-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_checkpoint(run_dir, requested):
    if requested is not None:
        path = requested if requested.is_absolute() else run_dir / requested
    else:
        path = run_dir / "checkpoints/best.pt"
        if not path.is_file():
            path = run_dir / "checkpoints/latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def _training_args(checkpoint, analysis_args):
    stored = dict(checkpoint.get("args") or {})
    if stored.get("model") != "ECalTpadContributorSetSlotModel":
        raise ValueError(
            "TPad ablation requires an ECalTpadContributorSetSlotModel checkpoint; "
            f"found {stored.get('model')!r}."
        )
    for key in PATH_ARGUMENTS:
        value = stored.get(key)
        if value is not None and not isinstance(value, Path):
            stored[key] = Path(value)
    if analysis_args.processed_cache_root is not None:
        stored["processed_cache_root"] = analysis_args.processed_cache_root
        stored["processed_cache"] = None
        stored["processed_source"] = None
    if analysis_args.batch_size is not None:
        stored["batch_size"] = analysis_args.batch_size
    stored["no_progress"] = True
    return SimpleNamespace(**stored)


def _feature_norm(checkpoint):
    stored = checkpoint.get("feature_norm")
    if stored is None:
        return None
    return {
        "first_continuous_col": int(stored["first_continuous_col"]),
        "mean": torch.as_tensor(stored["mean"], dtype=torch.float32),
        "std": torch.as_tensor(stored["std"], dtype=torch.float32),
    }


def restore_preprocessing(events, checkpoint, args, max_electrons):
    """Reapply canonical targets and the checkpoint's fixed normalization."""
    feature_norm = _feature_norm(checkpoint)
    hard_origin_rule = getattr(
        args,
        "hard_origin_target_rule",
        checkpoint_hard_origin_target_rule(checkpoint),
    )

    def transform(event):
        if not getattr(args, "supervise_noise", False):
            event = filter_noise_tensor_event(event)
        event = apply_variable_count_target_mode(
            event,
            valid_labels=tuple(args.valid_labels),
            target_mode=args.target_mode,
            max_electrons=max_electrons,
            hard_origin_target_rule=hard_origin_rule,
        )
        if feature_norm is not None:
            event = normalize_event_continuous_features(event, feature_norm)
        return event

    if not hasattr(events, "set_event_transform"):
        raise TypeError("Contributor-set ablation currently requires the sharded dataset.")
    events.set_event_transform(transform)
    return events


def remove_tpad_tokens(event):
    """Remove every TPad token while preserving ECal inputs and all targets."""
    if "tpad_mask" not in event:
        raise ValueError("TPad ablation requires event['tpad_mask'].")
    tpad_mask = event["tpad_mask"].to(dtype=torch.bool)
    if tpad_mask.shape != (event["x"].shape[0],):
        raise ValueError("event['tpad_mask'] must align with event['x'].")
    keep = ~tpad_mask
    ablated = dict(event)
    for key in ("x", "ecal_mask", "tpad_mask"):
        ablated[key] = event[key][keep]
    for key in ("tpad", "tpad_raw_pe"):
        if key in event:
            ablated[key] = event[key][:0]
    return ablated


class TPadAblatedDataset:
    """Read-only event view that removes TPad tokens after preprocessing."""

    def __init__(self, events):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return remove_tpad_tokens(self.events[index])

    def order_indices_for_access(self, indices, seed=None):
        return self.events.order_indices_for_access(indices, seed=seed)


def _strip_prefix(metrics, prefix):
    return {
        key[len(prefix) :]: value
        for key, value in metrics.items()
        if key.startswith(prefix)
    }


def summarize_ablation(reference, ablated):
    """Create unambiguous positive-is-helpful TPad deltas."""
    higher_is_better = (
        "accuracy",
        "origin_accuracy",
        "support_accuracy",
        "mixed_accuracy",
        "mixed_precision",
        "mixed_recall",
        "mixed_f1",
        "slot_accuracy",
        "slot_exact_accuracy",
        "count_accuracy",
        "count_accuracy_2e",
        "count_accuracy_3e",
    )
    lower_is_better = (
        "loss",
        "support_ce",
        "fraction_ce",
        "slot_bce",
        "fraction_mse",
        "fraction_mae",
        "raw_fraction_mae",
        "mixed_brier",
    )
    tpad_gain = {}
    for key in higher_is_better:
        if key in reference and key in ablated:
            tpad_gain[key] = float(reference[key]) - float(ablated[key])
    for key in lower_is_better:
        if key in reference and key in ablated:
            tpad_gain[f"{key}_reduction"] = float(ablated[key]) - float(reference[key])
    return {
        "reference": reference,
        "tpad_removed": ablated,
        "tpad_gain_positive_is_better": tpad_gain,
    }


def _crop_count_confusion(confusion):
    values = torch.as_tensor(confusion, dtype=torch.long)
    return values[2:4, 2:4].tolist()


def main(argv=None):
    analysis_args = parse_args(argv)
    if analysis_args.max_events is not None and analysis_args.max_events <= 0:
        raise ValueError("--max-events must be positive when provided.")
    if analysis_args.batch_size is not None and analysis_args.batch_size <= 0:
        raise ValueError("--batch-size must be positive when provided.")
    if analysis_args.max_plot_hits < 0:
        raise ValueError("--max-plot-hits must be non-negative.")

    run_dir = analysis_args.run_dir.resolve()
    checkpoint_path = resolve_checkpoint(run_dir, analysis_args.checkpoint)
    checkpoint = read_checkpoint(checkpoint_path, torch.device("cpu"))
    args = _training_args(checkpoint, analysis_args)
    output_dir = analysis_args.output_dir
    if output_dir is None:
        output_dir = run_dir / "tpad_ablation" / checkpoint_path.stem / analysis_args.split
    output_dir = output_dir.resolve()
    logger = setup_logging(
        output_dir,
        logger_name="contributor_set_tpad_ablation",
        log_filename="tpad_ablation.log",
    )
    device = resolve_device(analysis_args.device, logger)
    logger.info("Run directory: %s", run_dir)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Ablation: remove all TPad tokens; retain identical ECal tokens and targets")

    events, _sources, data_dir, _root_files = training.load_events(args, logger)
    if getattr(args, "supervise_noise", False):
        training.require_explicit_noise_targets(events)
    splits = checkpoint.get("splits")
    if not isinstance(splits, dict) or analysis_args.split not in splits:
        raise ValueError(f"Checkpoint has no saved {analysis_args.split!r} split.")
    indices = [int(index) for index in splits[analysis_args.split]]
    if indices and (min(indices) < 0 or max(indices) >= len(events)):
        raise ValueError("Saved split is incompatible with the loaded dataset.")
    if analysis_args.max_events is not None:
        indices = indices[: analysis_args.max_events]

    model_kwargs = dict(checkpoint.get("model_kwargs") or {})
    model = ECalTpadContributorSetSlotModel(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    events = restore_preprocessing(events, checkpoint, args, model.max_electrons)
    ablated_events = TPadAblatedDataset(events)

    evaluation_start = time.time()
    reference_metrics, reference_predictions, reference_plot_data = evaluate(
        model,
        events,
        indices,
        args,
        device,
        "reference",
        collect_predictions=True,
        max_plot_hits=analysis_args.max_plot_hits,
    )
    ablated_metrics, ablated_predictions, ablated_plot_data = evaluate(
        model,
        ablated_events,
        indices,
        args,
        device,
        "tpad_removed",
        collect_predictions=True,
        max_plot_hits=analysis_args.max_plot_hits,
    )
    elapsed = time.time() - evaluation_start
    reference = _strip_prefix(reference_metrics, "reference_")
    ablated = _strip_prefix(ablated_metrics, "tpad_removed_")
    summary = summarize_ablation(reference, ablated)
    summary.update(
        {
            "ablation_definition": "remove-all-tpad-tokens-v1",
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
            "split": analysis_args.split,
            "num_events": len(indices),
            "evaluation_elapsed_sec": elapsed,
        }
    )

    save_json(output_dir / "tpad_ablation_summary.json", summary)
    save_json(output_dir / "reference_event_predictions.json", reference_predictions)
    save_json(output_dir / "tpad_removed_event_predictions.json", ablated_predictions)
    torch.save(reference_plot_data, output_dir / "reference_hit_sample.pt")
    torch.save(ablated_plot_data, output_dir / "tpad_removed_hit_sample.pt")

    generated = []
    plot_specs = [
        (
            "electron_count_confusion_comparison.png",
            _crop_count_confusion(reference["count_confusion"]),
            _crop_count_confusion(ablated["count_confusion"]),
            ["2e", "3e"],
            "Derived electron-count confusion under TPad ablation",
        ),
        (
            "contributor_set_confusion_comparison.png",
            reference["support_confusion"],
            ablated["support_confusion"],
            contributor_set_labels(model.max_electrons),
            "Hit contributor-set confusion under TPad ablation",
        ),
        (
            "dominant_origin_confusion_comparison.png",
            reference["origin_confusion"],
            ablated["origin_confusion"],
            ["noise", "1", "2", "3"],
            "Derived dominant-origin confusion under TPad ablation",
        ),
        (
            "mixed_hit_confusion_comparison.png",
            reference["mixed_confusion"],
            ablated["mixed_confusion"],
            ["pure/noise", "mixed"],
            "Mixed-hit confusion under TPad ablation",
        ),
    ]
    for filename, reference_confusion, ablated_confusion, labels, title in plot_specs:
        path = output_dir / filename
        plot_confusion_ablation(
            reference_confusion,
            ablated_confusion,
            labels,
            path,
            title,
        )
        generated.append(path)

    task_path = output_dir / "task_metric_comparison.png"
    plot_task_metric_ablation(reference, ablated, task_path)
    generated.append(task_path)
    count_path = output_dir / "electron_count_tpad_ablation.png"
    plot_count_ablation(reference_predictions, ablated_predictions, count_path)
    generated.append(count_path)
    if reference_plot_data["fraction_target"].numel():
        fraction_path = output_dir / "fraction_mae_comparison.png"
        plot_fraction_mae_ablation(
            reference_plot_data,
            ablated_plot_data,
            fraction_path,
        )
        generated.append(fraction_path)
        for name, plot_data, title in (
            ("reference", reference_plot_data, "Mixed-hit probability with TPad"),
            ("tpad_removed", ablated_plot_data, "Mixed-hit probability with TPad removed"),
        ):
            path = output_dir / f"{name}_mixed_probability_diagnostics.png"
            plot_mixed_probability_diagnostics(
                plot_data["mixed_target"].numpy(),
                plot_data["mixed_probability"].numpy(),
                path,
                title=title,
            )
            generated.append(path)

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": analysis_args.split,
        "num_events": len(indices),
        "data_dir": str(data_dir),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "ablation_definition": "remove-all-tpad-tokens-v1",
        "generated_files": [
            "tpad_ablation_summary.json",
            "reference_event_predictions.json",
            "tpad_removed_event_predictions.json",
            "reference_hit_sample.pt",
            "tpad_removed_hit_sample.pt",
            *[str(path.relative_to(output_dir)) for path in generated],
        ],
    }
    save_json(output_dir / "tpad_ablation_manifest.json", manifest)
    gains = summary["tpad_gain_positive_is_better"]
    logger.info("Count accuracy with TPad: %.5f", reference["count_accuracy"])
    logger.info("Count accuracy without TPad: %.5f", ablated["count_accuracy"])
    logger.info("TPad count-accuracy gain: %+.5f", gains["count_accuracy"])
    logger.info("Saved paired TPad ablation to %s", output_dir)


if __name__ == "__main__":
    main()
