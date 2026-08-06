import time

import torch
import torch.nn.functional as F

from ml_ldmx.train.batching import chunks
from ml_ldmx.train.ecal_tpad_slot_batching import (
    collate_ecal_tpad_slot_batch,
    count_target_from_event,
    ecal_mask_from_event,
    fraction_targets_from_event,
    origin_targets_from_event,
    slot_targets_from_event,
)
from ml_ldmx.train.losses import soft_label_cross_entropy
from ml_ldmx.train.metrics import confusion_matrix_from_class_indices
from ml_ldmx.train.progress import make_progress


def count_cross_entropy_per_event(logits, target, args):
    """Return fixed-scale inverse-frequency-weighted count CE per event."""
    loss = F.cross_entropy(logits, target, reduction="none")
    count_weight = getattr(args, "count_class_weights", None)
    if count_weight is None:
        return loss

    count_weight = torch.as_tensor(
        count_weight,
        dtype=logits.dtype,
        device=logits.device,
    )
    if count_weight.shape != (logits.shape[-1],):
        raise ValueError(
            f"Expected {logits.shape[-1]} count class weights, got {count_weight.numel()}."
        )
    positive_weight = count_weight[count_weight > 0]
    if positive_weight.numel() == 0:
        raise ValueError("At least one count class weight must be positive.")
    # A fixed normalization keeps the loss scale independent of the classes in
    # one particular batch. This also makes weighting effective for the legacy
    # one-event helper, where PyTorch's weighted mean otherwise cancels it.
    return loss * count_weight[target] / positive_weight.mean()


def compute_event_losses(model, event: dict, device: torch.device, args):
    x = event["x"].to(device=device, dtype=torch.float32)
    ecal_mask = ecal_mask_from_event(event).to(device)
    outputs = model(x, ecal_mask=ecal_mask)

    origin_target = origin_targets_from_event(event, model.max_electrons).to(device)
    fraction_target = fraction_targets_from_event(
        event,
        origin_target.detach().cpu(),
        model.max_electrons,
    ).to(device)
    slot_target = slot_targets_from_event(
        event,
        origin_target,
        fraction_target,
        model.max_electrons,
    )
    count_target = count_target_from_event(event, slot_target, model.max_electrons).to(device)

    ecal_origin_logits = outputs["origin_logits"][ecal_mask]
    ecal_fraction_logits = outputs["fraction_logits"][ecal_mask]
    ecal_fraction_pred = outputs["fraction_pred"][ecal_mask]

    origin_weight = getattr(args, "origin_class_weights", None)
    if origin_weight is not None:
        origin_weight = torch.as_tensor(origin_weight, dtype=torch.float32, device=device)
    origin_loss = F.cross_entropy(ecal_origin_logits, origin_target, weight=origin_weight)
    fraction_loss = soft_label_cross_entropy(ecal_fraction_logits, fraction_target)
    slot_loss = F.binary_cross_entropy_with_logits(outputs["slot_valid_logits"], slot_target)
    count_loss = count_cross_entropy_per_event(
        outputs["count_logits"].unsqueeze(0),
        count_target.unsqueeze(0),
        args,
    )
    count_loss = count_loss.squeeze(0)
    total_loss = (
        args.lambda_origin * origin_loss
        + args.lambda_fraction * fraction_loss
        + args.lambda_slot * slot_loss
        + args.lambda_count * count_loss
    )

    pred_class = ecal_origin_logits.argmax(dim=1)
    fraction_abs_error = (ecal_fraction_pred - fraction_target).abs()
    slot_prob = torch.sigmoid(outputs["slot_valid_logits"])
    slot_pred = slot_prob > 0.5
    count_pred = outputs["count_logits"].argmax(dim=-1)
    slot_count_pred = slot_pred.to(dtype=torch.long).sum()

    return {
        "total_loss": total_loss,
        "origin_loss": origin_loss,
        "fraction_loss": fraction_loss,
        "slot_loss": slot_loss,
        "count_loss": count_loss,
        "fraction_mse": F.mse_loss(ecal_fraction_pred, fraction_target),
        "fraction_mae": fraction_abs_error.mean(),
        "per_hit_fraction_mae": fraction_abs_error.mean(dim=1),
        "fraction_target": fraction_target,
        "fraction_pred": ecal_fraction_pred,
        "pred_class": pred_class,
        "true_class": origin_target,
        "slot_target": slot_target,
        "slot_pred": slot_pred,
        "slot_prob": slot_prob,
        "count_target": count_target,
        "count_pred": count_pred,
        "slot_count_pred": slot_count_pred,
        "num_hits": origin_target.numel(),
    }


def compute_batch_losses(model, events: list[dict], device: torch.device, args):
    """Compute the existing per-event objective with one parallel padded forward."""
    batch = collate_ecal_tpad_slot_batch(events, model.max_electrons).to(device)
    outputs = model(
        batch.x,
        ecal_mask=batch.ecal_mask,
        key_padding_mask=~batch.valid_mask,
    )

    origin_weight = getattr(args, "origin_class_weights", None)
    if origin_weight is not None:
        origin_weight = torch.as_tensor(origin_weight, dtype=torch.float32, device=device)
    origin_token_loss = F.cross_entropy(
        outputs["origin_logits"].movedim(-1, 1),
        batch.origin_target,
        weight=origin_weight,
        ignore_index=-100,
        reduction="none",
    )
    if origin_weight is None:
        origin_denominator = batch.ecal_mask.sum(dim=1).to(dtype=origin_token_loss.dtype)
    else:
        safe_target = batch.origin_target.clamp_min(0)
        origin_denominator = (
            origin_weight[safe_target] * batch.ecal_mask.to(dtype=origin_token_loss.dtype)
        ).sum(dim=1)
    origin_loss_per_event = origin_token_loss.sum(dim=1) / origin_denominator

    fraction_token_loss = -(
        batch.fraction_target * F.log_softmax(outputs["fraction_logits"], dim=-1)
    ).sum(dim=-1)
    num_hits_per_event = batch.ecal_mask.sum(dim=1).to(dtype=fraction_token_loss.dtype)
    fraction_loss_per_event = fraction_token_loss.sum(dim=1) / num_hits_per_event

    slot_loss_per_event = F.binary_cross_entropy_with_logits(
        outputs["slot_valid_logits"],
        batch.slot_target,
        reduction="none",
    ).mean(dim=1)

    count_loss_per_event = count_cross_entropy_per_event(
        outputs["count_logits"],
        batch.count_target,
        args,
    )
    total_loss_per_event = (
        args.lambda_origin * origin_loss_per_event
        + args.lambda_fraction * fraction_loss_per_event
        + args.lambda_slot * slot_loss_per_event
        + args.lambda_count * count_loss_per_event
    )

    ecal_fraction_pred = outputs["fraction_pred"][batch.ecal_mask]
    fraction_target = batch.fraction_target[batch.ecal_mask]
    true_class = batch.origin_target[batch.ecal_mask]
    pred_class = outputs["origin_logits"][batch.ecal_mask].argmax(dim=1)
    fraction_abs_error = (ecal_fraction_pred - fraction_target).abs()
    slot_prob = torch.sigmoid(outputs["slot_valid_logits"])
    slot_pred = slot_prob > 0.5
    count_pred = outputs["count_logits"].argmax(dim=-1)
    slot_count_pred = slot_pred.to(dtype=torch.long).sum(dim=1)

    return {
        "total_loss": total_loss_per_event.mean(),
        "origin_loss": origin_loss_per_event.mean(),
        "fraction_loss": fraction_loss_per_event.mean(),
        "slot_loss": slot_loss_per_event.mean(),
        "count_loss": count_loss_per_event.mean(),
        "fraction_mse": F.mse_loss(ecal_fraction_pred, fraction_target),
        "fraction_mae": fraction_abs_error.mean(),
        "per_hit_fraction_mae": fraction_abs_error.mean(dim=1),
        "fraction_target": fraction_target,
        "fraction_pred": ecal_fraction_pred,
        "pred_class": pred_class,
        "true_class": true_class,
        "slot_target": batch.slot_target,
        "slot_pred": slot_pred,
        "slot_prob": slot_prob,
        "count_target": batch.count_target,
        "count_pred": count_pred,
        "slot_count_pred": slot_count_pred,
        "num_hits": batch.num_hits,
        "num_events": batch.batch_size,
        "batch": batch,
    }


def empty_slot_metric_totals(num_hit_classes: int, num_count_classes: int):
    return {
        "loss_sum": 0.0,
        "origin_loss_sum": 0.0,
        "fraction_loss_sum": 0.0,
        "slot_loss_sum": 0.0,
        "count_loss_sum": 0.0,
        "fraction_mse_sum": 0.0,
        "fraction_mae_sum": 0.0,
        "correct_hits": 0,
        "hits": 0,
        "events": 0,
        "slot_correct": 0,
        "slot_total": 0,
        "slot_exact_correct": 0,
        "count_correct": 0,
        "slot_count_correct": 0,
        "count_total_by_true": {idx: 0 for idx in range(num_count_classes)},
        "count_correct_by_true": {idx: 0 for idx in range(num_count_classes)},
        "hit_confusion": torch.zeros((num_hit_classes, num_hit_classes), dtype=torch.long),
        "count_confusion": torch.zeros((num_count_classes, num_count_classes), dtype=torch.long),
    }


def update_slot_metric_totals(totals: dict, losses: dict):
    num_hits = int(losses["num_hits"])
    num_events = int(losses.get("num_events", 1))
    loss_values = torch.stack(
        [
            losses["total_loss"],
            losses["origin_loss"],
            losses["fraction_loss"],
            losses["slot_loss"],
            losses["count_loss"],
            losses["fraction_mse"],
            losses["fraction_mae"],
        ]
    ).detach().cpu()
    totals["loss_sum"] += float(loss_values[0].item()) * num_events
    totals["origin_loss_sum"] += float(loss_values[1].item()) * num_events
    totals["fraction_loss_sum"] += float(loss_values[2].item()) * num_events
    totals["slot_loss_sum"] += float(loss_values[3].item()) * num_events
    totals["count_loss_sum"] += float(loss_values[4].item()) * num_events
    totals["fraction_mse_sum"] += float(loss_values[5].item()) * num_hits
    totals["fraction_mae_sum"] += float(loss_values[6].item()) * num_hits

    hit_confusion = confusion_matrix_from_class_indices(
        losses["true_class"],
        losses["pred_class"],
        totals["hit_confusion"].shape[0],
    ).cpu()
    totals["correct_hits"] += int(hit_confusion.diag().sum().item())
    totals["hits"] += num_hits
    totals["hit_confusion"] += hit_confusion

    slot_pred = losses["slot_pred"].detach().cpu()
    slot_true = losses["slot_target"].detach().cpu().to(dtype=torch.bool)
    if slot_pred.ndim == 1:
        slot_pred = slot_pred.unsqueeze(0)
        slot_true = slot_true.unsqueeze(0)
    totals["slot_correct"] += int((slot_pred == slot_true).sum().item())
    totals["slot_total"] += int(slot_true.numel())
    totals["slot_exact_correct"] += int((slot_pred == slot_true).all(dim=1).sum().item())

    count_true = losses["count_target"].detach().cpu().reshape(-1).to(dtype=torch.long)
    count_pred = losses["count_pred"].detach().cpu().reshape(-1).to(dtype=torch.long)
    slot_count_pred = slot_pred.to(dtype=torch.long).sum(dim=1)
    totals["count_correct"] += int((count_pred == count_true).sum().item())
    totals["slot_count_correct"] += int((slot_count_pred == count_true).sum().item())
    for true_value, pred_value in zip(count_true.tolist(), count_pred.tolist()):
        totals["count_total_by_true"][true_value] = (
            totals["count_total_by_true"].get(true_value, 0) + 1
        )
        totals["count_correct_by_true"][true_value] = (
            totals["count_correct_by_true"].get(true_value, 0)
            + int(pred_value == true_value)
        )
    totals["count_confusion"] += confusion_matrix_from_class_indices(
        count_true,
        count_pred,
        totals["count_confusion"].shape[0],
    ).cpu()
    totals["events"] += num_events


def finalize_slot_metrics(totals: dict, prefix: str = ""):
    events = max(1, totals["events"])
    hits = max(1, totals["hits"])
    slot_total = max(1, totals["slot_total"])
    metrics = {
        f"{prefix}loss": totals["loss_sum"] / events,
        f"{prefix}origin_ce": totals["origin_loss_sum"] / events,
        f"{prefix}fraction_ce": totals["fraction_loss_sum"] / events,
        f"{prefix}slot_bce": totals["slot_loss_sum"] / events,
        f"{prefix}count_ce": totals["count_loss_sum"] / events,
        f"{prefix}fraction_mse": totals["fraction_mse_sum"] / hits,
        f"{prefix}fraction_mae": totals["fraction_mae_sum"] / hits,
        f"{prefix}accuracy": totals["correct_hits"] / hits,
        f"{prefix}slot_accuracy": totals["slot_correct"] / slot_total,
        f"{prefix}slot_exact_accuracy": totals["slot_exact_correct"] / events,
        f"{prefix}count_accuracy": totals["count_correct"] / events,
        f"{prefix}slot_count_accuracy": totals["slot_count_correct"] / events,
        f"{prefix}num_hits": totals["hits"],
        f"{prefix}num_events": totals["events"],
    }
    for count, total in sorted(totals["count_total_by_true"].items()):
        if total > 0:
            metrics[f"{prefix}count_accuracy_{count}e"] = (
                totals["count_correct_by_true"].get(count, 0) / total
            )
    return metrics


def event_prediction_record(event_idx: int, event: dict, losses: dict) -> dict:
    raw_event_id = event.get("event_idx", event_idx)
    if isinstance(raw_event_id, torch.Tensor):
        raw_event_id = int(raw_event_id.detach().cpu().reshape(-1)[0].item())
    record = {
        "event_index": int(event_idx),
        "event_id": int(raw_event_id),
        "true_count": int(losses["count_target"].detach().cpu().item()),
        "predicted_count": int(losses["count_pred"].detach().cpu().item()),
        "predicted_slot_count": int(losses["slot_count_pred"].detach().cpu().item()),
        "slot_target": [float(value) for value in losses["slot_target"].detach().cpu().tolist()],
        "slot_probability": [float(value) for value in losses["slot_prob"].detach().cpu().tolist()],
    }
    for key in ("source_file", "source_entry", "source_label"):
        if key in event:
            value = event[key]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().reshape(-1)[0].item())
            record[key] = value
    return record


def batch_prediction_records(event_indices, events: list[dict], losses: dict) -> list[dict]:
    """Build event-level records from a batched result without reloading events."""
    records = []
    for row, (event_idx, event) in enumerate(zip(event_indices, events)):
        row_losses = {
            "count_target": losses["count_target"][row],
            "count_pred": losses["count_pred"][row],
            "slot_count_pred": losses["slot_count_pred"][row],
            "slot_target": losses["slot_target"][row],
            "slot_prob": losses["slot_prob"][row],
        }
        records.append(event_prediction_record(event_idx, event, row_losses))
    return records


def train_one_epoch(model, events, train_indices, optimizer, args, device, epoch, logger):
    model.train()
    if hasattr(events, "balanced_batches_for_access"):
        batch_indices = events.balanced_batches_for_access(
            train_indices,
            batch_size=args.batch_size,
            seed=args.seed + epoch,
        )
    elif hasattr(events, "order_indices_for_access"):
        shuffled_indices = events.order_indices_for_access(train_indices, seed=args.seed + epoch)
        batch_indices = list(chunks(shuffled_indices, args.batch_size))
    else:
        generator = torch.Generator().manual_seed(args.seed + epoch)
        shuffled_indices = [
            train_indices[idx]
            for idx in torch.randperm(len(train_indices), generator=generator).tolist()
        ]
        batch_indices = list(chunks(shuffled_indices, args.batch_size))
    totals = empty_slot_metric_totals(
        num_hit_classes=args.max_electrons + 1,
        num_count_classes=args.max_electrons + 1,
    )
    start_time = time.time()
    progress = make_progress(
        batch_indices,
        total=len(batch_indices),
        desc=f"epoch {epoch + 1}/{args.epochs} train",
        disable=args.no_progress,
        unit="batch",
    )

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        batch_events = [events[event_idx] for event_idx in batch]
        losses = compute_batch_losses(model, batch_events, device, args)
        update_slot_metric_totals(totals, losses)
        losses["total_loss"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if hasattr(progress, "set_postfix"):
            metrics = finalize_slot_metrics(totals)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                hit_acc=f"{metrics['accuracy']:.3f}",
                count_acc=f"{metrics['count_accuracy']:.3f}",
            )

    metrics = finalize_slot_metrics(totals, prefix="train_")
    metrics["train_elapsed_sec"] = time.time() - start_time
    logger.info(
        (
            "epoch=%03d train_loss=%.5f train_hit_acc=%.4f "
            "train_count_acc=%.4f train_fraction_mae=%.5f elapsed=%.1fs"
        ),
        epoch + 1,
        metrics["train_loss"],
        metrics["train_accuracy"],
        metrics["train_count_accuracy"],
        metrics["train_fraction_mae"],
        metrics["train_elapsed_sec"],
    )
    return metrics
