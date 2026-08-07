"""Training objective and metrics for the contributor-set slot model."""

import time

import torch
import torch.nn.functional as F

from ml_ldmx.eval.contributor_set_postprocessing import (
    fraction_targets_to_support,
    postprocess_contributor_set_outputs,
)
from ml_ldmx.train.batching import chunks
from ml_ldmx.train.ecal_tpad_slot_batching import collate_ecal_tpad_slot_batch
from ml_ldmx.train.metrics import confusion_matrix_from_class_indices
from ml_ldmx.train.progress import make_progress


def _fixed_scale_class_weight(target, weights, num_classes, dtype, device):
    if weights is None:
        return torch.ones_like(target, dtype=dtype, device=device)
    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    if weights.shape != (num_classes,):
        raise ValueError(f"Expected {num_classes} support weights, got {weights.numel()}.")
    positive = weights[weights > 0]
    if positive.numel() == 0:
        raise ValueError("At least one support-class weight must be positive.")
    return weights[target] / positive.mean()


def compute_batch_losses(model, events: list[dict], device: torch.device, args):
    """Compute the complete parallel objective for one padded event batch."""
    batch = collate_ecal_tpad_slot_batch(events, model.max_electrons).to(device)
    raw = model(
        batch.x,
        ecal_mask=batch.ecal_mask,
        key_padding_mask=~batch.valid_mask,
    )
    support_target = fraction_targets_to_support(
        batch.fraction_target,
        contribution_epsilon=args.contribution_epsilon,
    )

    support_token_loss = F.cross_entropy(
        raw["support_logits"].movedim(-1, 1),
        support_target,
        reduction="none",
    )
    support_scale = _fixed_scale_class_weight(
        support_target,
        getattr(args, "support_class_weights", None),
        model.num_support_classes,
        support_token_loss.dtype,
        device,
    )
    support_token_loss = support_token_loss * support_scale * batch.ecal_mask
    num_hits_per_event = batch.ecal_mask.sum(dim=1).to(dtype=support_token_loss.dtype).clamp_min(1.0)
    support_loss_per_event = support_token_loss.sum(dim=1) / num_hits_per_event

    fraction_token_loss = -(
        batch.fraction_target * F.log_softmax(raw["fraction_logits"], dim=-1)
    ).sum(dim=-1)
    fraction_loss_per_event = (
        fraction_token_loss * batch.ecal_mask
    ).sum(dim=1) / num_hits_per_event
    slot_loss_per_event = F.binary_cross_entropy_with_logits(
        raw["slot_valid_logits"],
        batch.slot_target,
        reduction="none",
    ).mean(dim=1)
    total_loss_per_event = (
        args.lambda_support * support_loss_per_event
        + args.lambda_fraction * fraction_loss_per_event
        + args.lambda_slot * slot_loss_per_event
    )

    processed = postprocess_contributor_set_outputs(
        raw,
        min_electrons=model.min_electrons,
        key_padding_mask=~batch.valid_mask,
    )
    ecal = batch.ecal_mask
    fraction_target = batch.fraction_target[ecal]
    fraction_pred = processed["fraction_prediction"][ecal]
    raw_fraction_pred = raw["raw_fraction_pred"][ecal]
    support_true = support_target[ecal]
    support_pred = processed["support_prediction"][ecal]
    true_origin = fraction_target.argmax(dim=-1)
    pred_origin = processed["dominant_origin"][ecal]
    support_cardinality_true = (
        (support_true.unsqueeze(-1).bitwise_and(
            1 << torch.arange(model.max_electrons, device=device)
        ) != 0).sum(dim=-1)
    )
    mixed_true = support_cardinality_true >= 2
    mixed_pred = processed["mixed_prediction"][ecal]
    mixed_probability = processed["mixed_probability"][ecal]
    fraction_abs_error = (fraction_pred - fraction_target).abs()
    raw_fraction_abs_error = (raw_fraction_pred - fraction_target).abs()

    return {
        "total_loss": total_loss_per_event.mean(),
        "support_loss": support_loss_per_event.mean(),
        "fraction_loss": fraction_loss_per_event.mean(),
        "slot_loss": slot_loss_per_event.mean(),
        "fraction_mse": F.mse_loss(fraction_pred, fraction_target),
        "fraction_mae": fraction_abs_error.mean(),
        "raw_fraction_mae": raw_fraction_abs_error.mean(),
        "per_hit_fraction_mae": fraction_abs_error.mean(dim=-1),
        "fraction_target": fraction_target,
        "fraction_pred": fraction_pred,
        "raw_fraction_pred": raw_fraction_pred,
        "support_target": support_true,
        "support_pred": support_pred,
        "support_probability": processed["support_probability"][ecal],
        "true_class": true_origin,
        "pred_class": pred_origin,
        "mixed_target": mixed_true,
        "mixed_pred": mixed_pred,
        "mixed_probability": mixed_probability,
        "slot_target": batch.slot_target,
        "slot_pred": processed["slot_valid_mask"],
        "slot_probability": processed["slot_probability"],
        "count_target": batch.count_target,
        "count_pred": processed["predicted_count"],
        "count_probability": processed["count_probability"],
        "count_values": processed["count_values"],
        "num_hits": batch.num_hits,
        "num_events": batch.batch_size,
        "batch": batch,
    }


def empty_metric_totals(num_fraction_classes: int, num_support_classes: int):
    return {
        "loss_sum": 0.0,
        "support_loss_sum": 0.0,
        "fraction_loss_sum": 0.0,
        "slot_loss_sum": 0.0,
        "fraction_mse_sum": 0.0,
        "fraction_mae_sum": 0.0,
        "raw_fraction_mae_sum": 0.0,
        "hits": 0,
        "events": 0,
        "origin_correct": 0,
        "support_correct": 0,
        "mixed_correct": 0,
        "mixed_brier_sum": 0.0,
        "slot_correct": 0,
        "slot_total": 0,
        "slot_exact_correct": 0,
        "count_correct": 0,
        "count_total_by_true": {},
        "count_correct_by_true": {},
        "origin_confusion": torch.zeros(
            (num_fraction_classes, num_fraction_classes), dtype=torch.long
        ),
        "support_confusion": torch.zeros(
            (num_support_classes, num_support_classes), dtype=torch.long
        ),
        "mixed_confusion": torch.zeros((2, 2), dtype=torch.long),
        "count_confusion": torch.zeros(
            (num_fraction_classes, num_fraction_classes), dtype=torch.long
        ),
    }


def update_metric_totals(totals: dict, losses: dict):
    num_hits = int(losses["num_hits"])
    num_events = int(losses["num_events"])
    scalars = torch.stack(
        [
            losses["total_loss"],
            losses["support_loss"],
            losses["fraction_loss"],
            losses["slot_loss"],
            losses["fraction_mse"],
            losses["fraction_mae"],
            losses["raw_fraction_mae"],
        ]
    ).detach().cpu()
    totals["loss_sum"] += float(scalars[0]) * num_events
    totals["support_loss_sum"] += float(scalars[1]) * num_events
    totals["fraction_loss_sum"] += float(scalars[2]) * num_events
    totals["slot_loss_sum"] += float(scalars[3]) * num_events
    totals["fraction_mse_sum"] += float(scalars[4]) * num_hits
    totals["fraction_mae_sum"] += float(scalars[5]) * num_hits
    totals["raw_fraction_mae_sum"] += float(scalars[6]) * num_hits

    origin_confusion = confusion_matrix_from_class_indices(
        losses["true_class"], losses["pred_class"], totals["origin_confusion"].shape[0]
    ).cpu()
    support_confusion = confusion_matrix_from_class_indices(
        losses["support_target"],
        losses["support_pred"],
        totals["support_confusion"].shape[0],
    ).cpu()
    mixed_confusion = confusion_matrix_from_class_indices(
        losses["mixed_target"].to(dtype=torch.long),
        losses["mixed_pred"].to(dtype=torch.long),
        2,
    ).cpu()
    totals["origin_confusion"] += origin_confusion
    totals["support_confusion"] += support_confusion
    totals["mixed_confusion"] += mixed_confusion
    totals["origin_correct"] += int(origin_confusion.diag().sum())
    totals["support_correct"] += int(support_confusion.diag().sum())
    totals["mixed_correct"] += int(mixed_confusion.diag().sum())
    mixed_target_float = losses["mixed_target"].to(dtype=losses["mixed_probability"].dtype)
    totals["mixed_brier_sum"] += float(
        ((losses["mixed_probability"] - mixed_target_float) ** 2).sum().detach().cpu()
    )

    slot_pred = losses["slot_pred"].detach().cpu().to(dtype=torch.bool)
    slot_true = losses["slot_target"].detach().cpu().to(dtype=torch.bool)
    totals["slot_correct"] += int((slot_pred == slot_true).sum())
    totals["slot_total"] += int(slot_true.numel())
    totals["slot_exact_correct"] += int((slot_pred == slot_true).all(dim=-1).sum())

    count_true = losses["count_target"].detach().cpu().to(dtype=torch.long)
    count_pred = losses["count_pred"].detach().cpu().to(dtype=torch.long)
    totals["count_correct"] += int((count_true == count_pred).sum())
    totals["count_confusion"] += confusion_matrix_from_class_indices(
        count_true, count_pred, totals["count_confusion"].shape[0]
    ).cpu()
    for true_value, pred_value in zip(count_true.tolist(), count_pred.tolist()):
        totals["count_total_by_true"][true_value] = totals["count_total_by_true"].get(true_value, 0) + 1
        totals["count_correct_by_true"][true_value] = (
            totals["count_correct_by_true"].get(true_value, 0) + int(true_value == pred_value)
        )

    totals["hits"] += num_hits
    totals["events"] += num_events


def finalize_metrics(totals: dict, prefix: str = ""):
    hits = max(1, totals["hits"])
    events = max(1, totals["events"])
    mixed_confusion = totals["mixed_confusion"]
    true_positive = int(mixed_confusion[1, 1])
    false_positive = int(mixed_confusion[0, 1])
    false_negative = int(mixed_confusion[1, 0])
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    origin_accuracy = totals["origin_correct"] / hits
    metrics = {
        f"{prefix}loss": totals["loss_sum"] / events,
        f"{prefix}support_ce": totals["support_loss_sum"] / events,
        f"{prefix}fraction_ce": totals["fraction_loss_sum"] / events,
        f"{prefix}slot_bce": totals["slot_loss_sum"] / events,
        f"{prefix}fraction_mse": totals["fraction_mse_sum"] / hits,
        f"{prefix}fraction_mae": totals["fraction_mae_sum"] / hits,
        f"{prefix}raw_fraction_mae": totals["raw_fraction_mae_sum"] / hits,
        f"{prefix}origin_accuracy": origin_accuracy,
        # Keep the conventional accuracy key so shared history plotting and
        # run-comparison tools can treat the derived dominant origin like the
        # hard-origin prediction of the maintained baselines.
        f"{prefix}accuracy": origin_accuracy,
        f"{prefix}support_accuracy": totals["support_correct"] / hits,
        f"{prefix}mixed_accuracy": totals["mixed_correct"] / hits,
        f"{prefix}mixed_precision": precision,
        f"{prefix}mixed_recall": recall,
        f"{prefix}mixed_f1": 2 * precision * recall / max(1e-12, precision + recall),
        f"{prefix}mixed_brier": totals["mixed_brier_sum"] / hits,
        f"{prefix}slot_accuracy": totals["slot_correct"] / max(1, totals["slot_total"]),
        f"{prefix}slot_exact_accuracy": totals["slot_exact_correct"] / events,
        f"{prefix}count_accuracy": totals["count_correct"] / events,
        f"{prefix}num_hits": totals["hits"],
        f"{prefix}num_events": totals["events"],
    }
    for count, total in sorted(totals["count_total_by_true"].items()):
        metrics[f"{prefix}count_accuracy_{count}e"] = (
            totals["count_correct_by_true"].get(count, 0) / total
        )
    return metrics


def batch_event_prediction_records(event_indices, events, losses):
    records = []
    for row, (event_idx, event) in enumerate(zip(event_indices, events)):
        raw_event_id = event.get("event_idx", event_idx)
        if isinstance(raw_event_id, torch.Tensor):
            raw_event_id = int(raw_event_id.detach().cpu().reshape(-1)[0])
        record = {
            "event_index": int(event_idx),
            "event_id": int(raw_event_id),
            "true_count": int(losses["count_target"][row].detach().cpu()),
            "predicted_count": int(losses["count_pred"][row].detach().cpu()),
            "slot_target": losses["slot_target"][row].detach().cpu().tolist(),
            "slot_probability": losses["slot_probability"][row].detach().cpu().tolist(),
        }
        for key in ("source_file", "source_entry", "source_label"):
            if key in event:
                value = event[key]
                if isinstance(value, torch.Tensor):
                    value = int(value.detach().cpu().reshape(-1)[0])
                record[key] = value
        records.append(record)
    return records


def train_one_epoch(model, events, train_indices, optimizer, args, device, epoch, logger):
    model.train()
    if hasattr(events, "balanced_batches_for_access"):
        batches = events.balanced_batches_for_access(
            train_indices, batch_size=args.batch_size, seed=args.seed + epoch
        )
    elif hasattr(events, "order_indices_for_access"):
        ordered = events.order_indices_for_access(train_indices, seed=args.seed + epoch)
        batches = list(chunks(ordered, args.batch_size))
    else:
        generator = torch.Generator().manual_seed(args.seed + epoch)
        ordered = [
            train_indices[index]
            for index in torch.randperm(len(train_indices), generator=generator).tolist()
        ]
        batches = list(chunks(ordered, args.batch_size))

    totals = empty_metric_totals(model.num_fraction_classes, model.num_support_classes)
    start = time.time()
    progress = make_progress(
        batches,
        total=len(batches),
        desc=f"epoch {epoch + 1}/{args.epochs} train",
        disable=args.no_progress,
        unit="batch",
    )
    for indices in progress:
        optimizer.zero_grad(set_to_none=True)
        batch_events = [events[index] for index in indices]
        losses = compute_batch_losses(model, batch_events, device, args)
        update_metric_totals(totals, losses)
        losses["total_loss"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if hasattr(progress, "set_postfix"):
            current = finalize_metrics(totals)
            progress.set_postfix(
                loss=f"{current['loss']:.4f}",
                origin=f"{current['origin_accuracy']:.3f}",
                mixed=f"{current['mixed_f1']:.3f}",
                count=f"{current['count_accuracy']:.3f}",
            )

    metrics = finalize_metrics(totals, prefix="train_")
    metrics["train_elapsed_sec"] = time.time() - start
    logger.info(
        "epoch=%03d train_loss=%.5f origin_acc=%.4f mixed_f1=%.4f count_acc=%.4f elapsed=%.1fs",
        epoch + 1,
        metrics["train_loss"],
        metrics["train_origin_accuracy"],
        metrics["train_mixed_f1"],
        metrics["train_count_accuracy"],
        metrics["train_elapsed_sec"],
    )
    return metrics
