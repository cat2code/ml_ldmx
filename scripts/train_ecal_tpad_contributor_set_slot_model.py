#!/usr/bin/env python3
"""Train the parallel MLPF-inspired contributor-set slot model.

This entry point deliberately does not replace ``train_ecal_tpad_slot_model.py``.
It reuses its data preparation only; the model, objectives, postprocessing,
metrics, checkpoints, and plots have distinct names and output directories.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# These functions are the established canonical-y/noise/data-cache pipeline.
# Importing them avoids maintaining a second interpretation of the same cache.
from train_ecal_tpad_slot_model import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROCESSED_DIR,
    load_events,
    make_scheduler,
    prepare_targets_and_features,
    require_explicit_noise_targets,
    save_event_count_plots,
)

from ml_ldmx.datasets.stats import target_order_counts  # noqa: E402
from ml_ldmx.datasets.tensorize import DOMINANT_ORIGIN_TARGET_RULE  # noqa: E402
from ml_ldmx.eval.contributor_set_postprocessing import (  # noqa: E402
    fraction_targets_to_support,
    postprocess_contributor_set_outputs,
)
from ml_ldmx.eval.ecal_tpad_contributor_set_slot_model import evaluate  # noqa: E402
from ml_ldmx.io.artifacts import save_config, save_history, save_json  # noqa: E402
from ml_ldmx.models import ECalTpadContributorSetSlotModel  # noqa: E402
from ml_ldmx.train.checkpoints import (  # noqa: E402
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from ml_ldmx.train.early_stopping import early_stopping_state_from_history  # noqa: E402
from ml_ldmx.train.ecal_tpad_contributor_set_slot_model import train_one_epoch  # noqa: E402
from ml_ldmx.train.ecal_tpad_slot_batching import (  # noqa: E402
    ecal_mask_from_event,
    fraction_targets_from_event,
    origin_targets_from_event,
)
from ml_ldmx.train.logging import setup_logging  # noqa: E402
from ml_ldmx.train.modeling import count_trainable_parameters  # noqa: E402
from ml_ldmx.train.paths import resolve_run_dir  # noqa: E402
from ml_ldmx.train.splits import deterministic_split  # noqa: E402
from ml_ldmx.train.utils import resolve_device  # noqa: E402
from ml_ldmx.viz.contributor_sets import (  # noqa: E402
    contributor_set_labels,
    plot_contributor_set_history,
    plot_fraction_reconstruction,
    plot_mixed_probability_diagnostics,
)
from ml_ldmx.viz.ecal import plot_ecal_truth_prediction_pair  # noqa: E402
from ml_ldmx.viz.training import plot_confusion_matrix, plot_history  # noqa: E402


DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data/processed/production_10M_001_sharded"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/ecal_tpad_contributor_set_slot_model"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train ECalTpadContributorSetSlotModel on balanced 2e/3e ECal + TPad events."
        )
    )
    parser.add_argument("--processed-cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--events-per-source",
        type=int,
        default=5_000,
        help="Events drawn from each of the 2e and 3e caches (total is twice this value).",
    )
    parser.add_argument("--shard-cache-size", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-electrons", type=int, default=3)
    parser.add_argument("--min-electrons", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-type-embedding", action="store_true")
    parser.add_argument("--lambda-support", type=float, default=1.0)
    parser.add_argument("--lambda-fraction", type=float, default=1.0)
    parser.add_argument("--lambda-slot", type=float, default=1.0)
    parser.add_argument(
        "--contribution-epsilon",
        type=float,
        default=0.0,
        help="A truth fraction must exceed this value to belong to the contributor set.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="plateau")
    parser.add_argument("--plateau-patience", type=int, default=1)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--ecal-energy-transform", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--tpad-pe-transform", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--num-ecal-plots", type=int, default=2)
    parser.add_argument("--max-plot-hits", type=int, default=100_000)
    args = parser.parse_args(argv)

    # Attributes consumed by the shared, already-tested data-loading functions.
    args.events_per_class = 10
    args.max_events = None
    args.data_root = DEFAULT_DATA_ROOT
    args.processed_dir = DEFAULT_PROCESSED_DIR
    args.processed_cache = None
    args.processed_source = None
    args.force_sharded_cache = False
    args.allow_incomplete_sharded_cache = False
    args.max_cache_root_files = None
    args.max_events_per_root_file = None
    args.output_dir = args.output_root
    args.valid_labels = [1, 2, 3]
    args.target_mode = "canonical-y"
    args.keep_noise = False
    args.supervise_noise = True
    args.event_log_every = 0
    args.read_step_size = 500
    args.allow_fewer_events = False
    args.hard_origin_target_rule = DOMINANT_ORIGIN_TARGET_RULE
    args.model = "ECalTpadContributorSetSlotModel"
    args.postprocessing = "legal-prefix-contributor-set-gated-fractions-v1"
    args.support_target_rule = "positive-truth-fraction-bit-mask-v1"
    args.support_weighting = "sqrt-inverse-frequency-v1"
    return args


def validate_args(args):
    positive_integer_names = (
        "events_per_source",
        "shard_cache_size",
        "epochs",
        "early_stopping_min_epochs",
        "batch_size",
        "checkpoint_every",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "max_electrons",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.early_stopping_min_epochs > args.epochs:
        raise ValueError("--early-stopping-min-epochs cannot exceed --epochs.")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping patience and minimum delta must be non-negative.")
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError("--hidden-dim must be divisible by --num-heads.")
    if not 0 <= args.min_electrons <= args.max_electrons:
        raise ValueError("--min-electrons must be in 0..--max-electrons.")
    if args.max_electrons != 3 or args.min_electrons != 2:
        raise ValueError("The current production experiment is defined for 2e/3e data (min=2, max=3).")
    if args.contribution_epsilon < 0:
        raise ValueError("--contribution-epsilon must be non-negative.")
    if min(args.lambda_support, args.lambda_fraction, args.lambda_slot) < 0:
        raise ValueError("Loss coefficients must be non-negative.")
    if args.lambda_support + args.lambda_fraction + args.lambda_slot <= 0:
        raise ValueError("At least one loss coefficient must be positive.")
    if args.max_plot_hits < 0 or args.num_ecal_plots < 0:
        raise ValueError("Plot limits must be non-negative.")


def model_kwargs_from_args(args, input_dim):
    return {
        "in_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "max_electrons": args.max_electrons,
        "min_electrons": args.min_electrons,
        "dropout": args.dropout,
        "use_type_embedding": not args.no_type_embedding,
    }


def _sqrt_inverse_frequency_weights(counts, num_classes):
    weights = [0.0] * num_classes
    positive = {index: count for index, count in counts.items() if count > 0}
    if not positive:
        return [1.0] * num_classes
    total = sum(positive.values())
    for index, count in positive.items():
        weights[index] = math.sqrt(total / (len(positive) * count))
    mean = sum(value for value in weights if value > 0) / len(positive)
    return [value / mean if value > 0 else 0.0 for value in weights]


def add_support_class_weights(args, events, train_indices, logger):
    """Fit a tempered class balance from exact fraction-support targets."""
    counts = Counter()
    ordered = (
        events.order_indices_for_access(train_indices)
        if hasattr(events, "order_indices_for_access")
        else train_indices
    )
    for event_index in ordered:
        event = events[event_index]
        origin = origin_targets_from_event(event, args.max_electrons)
        fractions = fraction_targets_from_event(event, origin, args.max_electrons)
        support = fraction_targets_to_support(
            fractions,
            contribution_epsilon=args.contribution_epsilon,
        )
        counts.update(int(value) for value in support.tolist())
    args.support_class_counts = {
        str(index): int(counts.get(index, 0)) for index in range(1 << args.max_electrons)
    }
    args.support_class_weights = _sqrt_inverse_frequency_weights(
        counts,
        1 << args.max_electrons,
    )
    logger.info("Contributor-set counts: %s", args.support_class_counts)
    logger.info("Contributor-set sqrt-inverse weights: %s", args.support_class_weights)


@torch.no_grad()
def save_ecal_examples(model, events, indices, args, device, run_dir):
    if args.num_ecal_plots <= 0:
        return
    model.eval()
    for event_index in indices[: args.num_ecal_plots]:
        event = events[event_index]
        x = event["x"].to(device=device, dtype=torch.float32)
        ecal_mask = ecal_mask_from_event(event).to(device)
        raw = model(x, ecal_mask=ecal_mask)
        processed = postprocess_contributor_set_outputs(
            raw,
            min_electrons=model.min_electrons,
        )
        true_labels = origin_targets_from_event(event, model.max_electrons).cpu()
        pred_labels = processed["dominant_origin"][ecal_mask].cpu()
        plot_ecal_truth_prediction_pair(
            event["ecal_pos"].cpu(),
            true_labels,
            pred_labels,
            truth_path=run_dir / f"test_ecal_event_{event_index:04d}_truth.png",
            predicted_path=run_dir / f"test_ecal_event_{event_index:04d}_predicted.png",
            truth_title=f"Test event {event_index}: true dominant contributor",
            predicted_title=f"Test event {event_index}: reconstructed dominant contributor",
            labels=list(range(model.max_electrons + 1)),
        )


def save_diagnostic_plots(run_dir, history, test_metrics, test_predictions, plot_data, args):
    plot_history(history, run_dir, title_prefix="Contributor-set slot model")
    plot_contributor_set_history(history, run_dir / "task_history.png")
    plot_confusion_matrix(
        test_metrics["test_support_confusion"],
        contributor_set_labels(args.max_electrons),
        run_dir / "test_contributor_set_confusion_matrix.png",
        "Test hit contributor-set confusion matrix",
    )
    plot_confusion_matrix(
        test_metrics["test_origin_confusion"],
        list(range(args.max_electrons + 1)),
        run_dir / "test_derived_origin_confusion_matrix.png",
        "Test derived dominant-origin confusion matrix",
    )
    plot_confusion_matrix(
        test_metrics["test_mixed_confusion"],
        ["pure/noise", "mixed"],
        run_dir / "test_mixed_hit_confusion_matrix.png",
        "Test learned mixed-hit confusion matrix",
    )
    save_event_count_plots(run_dir, test_predictions, args)
    if plot_data["mixed_probability"].numel():
        plot_mixed_probability_diagnostics(
            plot_data["mixed_target"].numpy(),
            plot_data["mixed_probability"].numpy(),
            run_dir / "test_mixed_probability_diagnostics.png",
        )
        # Exclude the explicit noise column; the three panels correspond to
        # electron slots 1..3.
        plot_fraction_reconstruction(
            plot_data["fraction_target"][:, 1:].numpy(),
            plot_data["fraction_pred"][:, 1:].numpy(),
            run_dir / "test_fraction_scatter.png",
            electron_labels=tuple(args.valid_labels),
        )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    run_dir = resolve_run_dir(args)
    logger = setup_logging(run_dir, logger_name="ecal_tpad_contributor_set_slot_model")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device, logger)
    logger.info("Output directory: %s", run_dir)
    logger.info("Using device: %s", device)
    logger.info(
        "Model semantics: slot validity + contributor-set classification + fractions; "
        "count/origin/mixed are reconstructed, not independent heads"
    )

    events, event_sources, data_dir, root_files = load_events(args, logger)
    args.training_batch_policy = (
        "source-balanced-shard-local-v1"
        if hasattr(events, "balanced_batches_for_access")
        else "random-event-v1"
    )
    require_explicit_noise_targets(events)
    if len(events) < 20:
        raise ValueError(f"Need at least 20 events; loaded {len(events)}.")
    splits = deterministic_split(len(events), args.seed)
    logger.info(
        "Split sizes: train=%s val=%s test=%s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    feature_norm = prepare_targets_and_features(events, splits, args, logger)
    add_support_class_weights(args, events, splits["train"], logger)

    input_dim = int(events[0]["x"].shape[1])
    model_kwargs = model_kwargs_from_args(args, input_dim)
    model = ECalTpadContributorSetSlotModel(**model_kwargs).to(device)
    logger.info("Trainable model parameters: %s", count_trainable_parameters(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)

    history = []
    best_val_loss = float("inf")
    start_epoch = 0
    if args.resume is not None:
        checkpoint = read_checkpoint(args.resume, device)
        checkpoint_args = checkpoint.get("args", {})
        if checkpoint_args.get("model") != args.model:
            raise ValueError("--resume does not contain an ECalTpadContributorSetSlotModel run.")
        if checkpoint.get("model_kwargs") != model_kwargs:
            raise ValueError("Checkpoint architecture does not match the requested architecture.")
        if checkpoint.get("splits") != splits:
            raise ValueError("Checkpoint split does not match this data selection and seed.")
        semantic_keys = (
            "target_mode",
            "hard_origin_target_rule",
            "contribution_epsilon",
            "support_target_rule",
            "postprocessing",
            "ecal_energy_transform",
            "tpad_pe_transform",
            "training_batch_policy",
        )
        mismatches = {
            key: (checkpoint_args.get(key), getattr(args, key))
            for key in semantic_keys
            if checkpoint_args.get(key) != getattr(args, key)
        }
        if mismatches:
            raise ValueError(f"Checkpoint reconstruction semantics do not match: {mismatches}")
        restore_checkpoint(checkpoint, model, optimizer, scheduler)
        history = checkpoint.get("history", [])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from %s at epoch %s", args.resume, start_epoch + 1)

    save_config(
        args,
        run_dir,
        data_dir,
        root_files,
        event_sources,
        splits,
        {name: target_order_counts(events, indices) for name, indices in splits.items()},
    )
    save_history(history, run_dir)

    early_stopping = early_stopping_state_from_history(
        history,
        min_delta=args.early_stopping_min_delta,
    )
    stop_marker = run_dir / "STOP_AFTER_EPOCH"
    stop_reason = "max_epochs"
    interrupted = False
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_metrics = {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
            }
            epoch_metrics.update(
                train_one_epoch(
                    model,
                    events,
                    splits["train"],
                    optimizer,
                    args,
                    device,
                    epoch,
                    logger,
                )
            )
            validation_start = time.time()
            val_metrics, _predictions, _plot_data = evaluate(
                model, events, splits["val"], args, device, "val"
            )
            val_metrics["val_elapsed_sec"] = time.time() - validation_start
            epoch_metrics.update(val_metrics)
            early_stopping, significant = early_stopping.update(
                val_metrics["val_loss"], args.early_stopping_min_delta
            )
            epoch_metrics["early_stopping_significant_improvement"] = significant
            epoch_metrics["early_stopping_bad_epochs"] = early_stopping.bad_epochs
            history.append(epoch_metrics)
            logger.info(
                "epoch=%03d val_loss=%.5f support_acc=%.4f mixed_f1=%.4f count_acc=%.4f",
                epoch + 1,
                val_metrics["val_loss"],
                val_metrics["val_support_accuracy"],
                val_metrics["val_mixed_f1"],
                val_metrics["val_count_accuracy"],
            )
            if scheduler is not None:
                scheduler.step(val_metrics["val_loss"])
            save_history(history, run_dir)
            plot_history(history, run_dir, title_prefix="Contributor-set slot model")
            plot_contributor_set_history(history, run_dir / "task_history.png")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                save_checkpoint(
                    run_dir / "checkpoints/best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    history,
                    best_val_loss,
                    model_kwargs,
                    feature_norm,
                    splits,
                )
            save_checkpoint(
                run_dir / "checkpoints/latest.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                history,
                best_val_loss,
                model_kwargs,
                feature_norm,
                splits,
            )
            if (epoch + 1) % args.checkpoint_every == 0:
                save_checkpoint(
                    run_dir / f"checkpoints/epoch_{epoch + 1:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    history,
                    best_val_loss,
                    model_kwargs,
                    feature_norm,
                    splits,
                )
            if stop_marker.exists():
                stop_reason = "manual_stop_after_epoch"
                stop_marker.unlink(missing_ok=True)
                logger.info("Graceful stop requested after checkpointed epoch %s.", epoch + 1)
                break
            if early_stopping.should_stop(
                completed_epochs=epoch + 1,
                min_epochs=args.early_stopping_min_epochs,
                patience=args.early_stopping_patience,
            ):
                stop_reason = "early_stopping"
                logger.info("Early stopping after epoch %s.", epoch + 1)
                break
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "keyboard_interrupt"
        logger.warning("Interrupt received; saving resumable checkpoint.")
        save_checkpoint(
            run_dir / "checkpoints/interrupted_latest.pt",
            model,
            optimizer,
            scheduler,
            max(start_epoch - 1, len(history) - 1),
            args,
            history,
            best_val_loss,
            model_kwargs,
            feature_norm,
            splits,
        )
    finally:
        save_history(history, run_dir)
        test_start = time.time()
        test_metrics, test_predictions, plot_data = evaluate(
            model,
            events,
            splits["test"],
            args,
            device,
            "test",
            collect_predictions=True,
            max_plot_hits=args.max_plot_hits,
        )
        test_metrics["test_elapsed_sec"] = time.time() - test_start
        val_metrics, val_predictions, _unused = evaluate(
            model,
            events,
            splits["val"],
            args,
            device,
            "final_val",
            collect_predictions=True,
        )
        final_metrics = {
            "interrupted": interrupted,
            "stop_reason": stop_reason,
            "completed_epochs": len(history),
            "best_val_loss": best_val_loss,
            **val_metrics,
            **test_metrics,
        }
        save_json(run_dir / "final_metrics.json", final_metrics)
        save_json(run_dir / "test_event_predictions.json", test_predictions)
        save_json(run_dir / "val_event_predictions.json", val_predictions)
        torch.save(plot_data, run_dir / "test_hit_prediction_sample.pt")
        save_diagnostic_plots(run_dir, history, test_metrics, test_predictions, plot_data, args)
        save_ecal_examples(model, events, splits["test"], args, device, run_dir)
        save_checkpoint(
            run_dir / "checkpoints/latest.pt",
            model,
            optimizer,
            scheduler,
            max(start_epoch - 1, len(history) - 1),
            args,
            history,
            best_val_loss,
            model_kwargs,
            feature_norm,
            splits,
        )
        logger.info(
            "Final test: loss=%.5f support_acc=%.4f mixed_f1=%.4f count_acc=%.4f",
            test_metrics["test_loss"],
            test_metrics["test_support_accuracy"],
            test_metrics["test_mixed_f1"],
            test_metrics["test_count_accuracy"],
        )
        logger.info("Saved outputs to %s", run_dir)


if __name__ == "__main__":
    main()
