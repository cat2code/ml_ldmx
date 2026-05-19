import time

import torch
import torch.nn.functional as F

from mldmx.train.batching import chunks
from mldmx.train.losses import soft_label_cross_entropy
from mldmx.train.metrics import empty_metric_totals, finalize_metrics, update_metric_totals
from mldmx.train.progress import make_progress


def compute_event_losses(model, event, device, lambda_fraction):
    x = event["x"].to(device)
    ecal_mask = event["ecal_mask"].to(device)
    y = event["y"].to(device)
    fraction_target = event["fraction_target"].to(device)

    outputs = model(x)
    ecal_origin_logits = outputs["origin_logits"][ecal_mask]
    ecal_fraction_logits = outputs["fraction_logits"][ecal_mask]
    ecal_fraction_pred = outputs["fraction_pred"][ecal_mask]

    origin_loss = F.cross_entropy(ecal_origin_logits, y)
    fraction_loss = soft_label_cross_entropy(ecal_fraction_logits, fraction_target)
    total_loss = origin_loss + lambda_fraction * fraction_loss

    pred_class = ecal_origin_logits.argmax(dim=1)
    fraction_abs_error = (ecal_fraction_pred - fraction_target).abs()
    return {
        "total_loss": total_loss,
        "origin_loss": origin_loss,
        "fraction_loss": fraction_loss,
        "fraction_mse": F.mse_loss(ecal_fraction_pred, fraction_target),
        "fraction_mae": fraction_abs_error.mean(),
        "per_hit_fraction_mae": fraction_abs_error.mean(dim=1),
        "fraction_target": fraction_target,
        "fraction_pred": ecal_fraction_pred,
        "pred_class": pred_class,
        "true_class": y,
        "num_hits": y.numel(),
    }


def train_one_epoch(model, events, train_indices, optimizer, args, device, epoch, logger):
    model.train()
    generator = torch.Generator().manual_seed(args.seed + epoch)
    shuffled_indices = [
        train_indices[idx]
        for idx in torch.randperm(len(train_indices), generator=generator).tolist()
    ]
    batch_indices = list(chunks(shuffled_indices, args.batch_size))
    totals = empty_metric_totals(num_classes=len(args.valid_labels))
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
        batch_loss_sum = None
        batch_hits = 0
        for event_idx in batch:
            losses = compute_event_losses(model, events[event_idx], device, args.lambda_fraction)
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
        "epoch=%03d train_loss=%.5f train_acc=%.4f train_fraction_mae=%.5f elapsed=%.1fs",
        epoch + 1,
        metrics["train_loss"],
        metrics["train_accuracy"],
        metrics["train_fraction_mae"],
        metrics["train_elapsed_sec"],
    )
    return metrics
