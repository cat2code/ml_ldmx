"""Training utilities for maintained hit-origin classifier baselines."""

import time

import torch
import torch.nn.functional as F

from mldmx.train.batching import chunks
from mldmx.train.progress import make_progress


def compute_event_losses(model, event_or_view, view_fn, device):
    """Compute ECal-supervised classification outputs for one event or prepared view."""
    view = event_or_view if view_fn is None else view_fn(event_or_view)
    x = view["x"].to(device=device, dtype=torch.float32)
    ecal_mask = view["ecal_mask"].to(device=device, dtype=torch.bool)
    target = view["y"].to(device=device, dtype=torch.long)

    logits = model(x)
    supervised_logits = logits[ecal_mask]
    if supervised_logits.shape[0] != target.shape[0]:
        raise ValueError(
            "Baseline supervised logits and targets are not aligned: "
            f"{supervised_logits.shape[0]} vs {target.shape[0]}."
        )
    loss = F.cross_entropy(supervised_logits, target)
    pred_class = supervised_logits.argmax(dim=1)
    return {
        "total_loss": loss,
        "pred_class": pred_class,
        "true_class": target,
        "num_hits": target.numel(),
        "view": view,
    }


def empty_metric_totals(num_classes):
    return {
        "loss_sum": 0.0,
        "correct": 0,
        "hits": 0,
        "confusion": torch.zeros((num_classes, num_classes), dtype=torch.long),
    }


def update_metric_totals(totals, losses):
    num_hits = int(losses["num_hits"])
    totals["loss_sum"] += float(losses["total_loss"].detach().cpu().item()) * num_hits
    pred = losses["pred_class"].detach().cpu()
    true = losses["true_class"].detach().cpu()
    totals["correct"] += int((pred == true).sum().item())
    totals["hits"] += num_hits
    for true_idx, pred_idx in zip(true.tolist(), pred.tolist()):
        totals["confusion"][int(true_idx), int(pred_idx)] += 1


def finalize_metrics(totals, prefix=""):
    hits = max(1, totals["hits"])
    return {
        f"{prefix}loss": totals["loss_sum"] / hits,
        f"{prefix}accuracy": totals["correct"] / hits,
        f"{prefix}num_hits": totals["hits"],
        f"{prefix}confusion": totals["confusion"].tolist(),
    }


def train_one_epoch(model, events, train_indices, view_fn, optimizer, args, device, epoch, logger):
    model.train()
    if hasattr(events, "order_indices_for_access"):
        shuffled_indices = events.order_indices_for_access(train_indices, seed=args.seed + epoch)
    else:
        generator = torch.Generator().manual_seed(args.seed + epoch)
        shuffled_indices = [
            train_indices[idx]
            for idx in torch.randperm(len(train_indices), generator=generator).tolist()
        ]
    batches = list(chunks(shuffled_indices, args.batch_size))
    totals = empty_metric_totals(num_classes=len(args.valid_labels))
    start_time = time.time()
    progress = make_progress(
        batches,
        total=len(batches),
        desc=f"epoch {epoch + 1}/{args.epochs} train",
        disable=args.no_progress,
        unit="batch",
    )

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        batch_loss_sum = None
        batch_hits = 0
        for event_idx in batch:
            losses = compute_event_losses(model, events[event_idx], view_fn, device)
            update_metric_totals(totals, losses)
            weighted_loss = losses["total_loss"] * losses["num_hits"]
            batch_loss_sum = weighted_loss if batch_loss_sum is None else batch_loss_sum + weighted_loss
            batch_hits += int(losses["num_hits"])

        batch_loss = batch_loss_sum / max(1, batch_hits)
        batch_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if hasattr(progress, "set_postfix"):
            metrics = finalize_metrics(totals)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                acc=f"{metrics['accuracy']:.3f}",
            )

    metrics = finalize_metrics(totals, prefix="train_")
    metrics["train_elapsed_sec"] = time.time() - start_time
    logger.info(
        "epoch=%03d train_loss=%.5f train_acc=%.4f elapsed=%.1fs",
        epoch + 1,
        metrics["train_loss"],
        metrics["train_accuracy"],
        metrics["train_elapsed_sec"],
    )
    return metrics
