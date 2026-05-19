import torch


def empty_metric_totals(num_classes):
    return {
        "loss_sum": 0.0,
        "origin_loss_sum": 0.0,
        "fraction_loss_sum": 0.0,
        "fraction_mse_sum": 0.0,
        "fraction_mae_sum": 0.0,
        "correct": 0,
        "hits": 0,
        "confusion": torch.zeros((num_classes, num_classes), dtype=torch.long),
    }


def update_metric_totals(totals, losses):
    num_hits = int(losses["num_hits"])
    totals["loss_sum"] += float(losses["total_loss"].detach().cpu().item()) * num_hits
    totals["origin_loss_sum"] += float(losses["origin_loss"].detach().cpu().item()) * num_hits
    totals["fraction_loss_sum"] += float(losses["fraction_loss"].detach().cpu().item()) * num_hits
    totals["fraction_mse_sum"] += float(losses["fraction_mse"].detach().cpu().item()) * num_hits
    totals["fraction_mae_sum"] += float(losses["fraction_mae"].detach().cpu().item()) * num_hits
    pred = losses["pred_class"].detach().cpu()
    true = losses["true_class"].detach().cpu()
    totals["correct"] += int((pred == true).sum().item())
    totals["hits"] += num_hits
    for true_idx, pred_idx in zip(true.tolist(), pred.tolist()):
        totals["confusion"][int(true_idx), int(pred_idx)] += 1


def finalize_metrics(totals, prefix=""):
    hits = max(1, totals["hits"])
    metrics = {
        f"{prefix}loss": totals["loss_sum"] / hits,
        f"{prefix}origin_ce": totals["origin_loss_sum"] / hits,
        f"{prefix}fraction_ce": totals["fraction_loss_sum"] / hits,
        f"{prefix}fraction_mse": totals["fraction_mse_sum"] / hits,
        f"{prefix}fraction_mae": totals["fraction_mae_sum"] / hits,
        f"{prefix}accuracy": totals["correct"] / hits,
        f"{prefix}num_hits": totals["hits"],
    }
    class_metrics = classification_metrics_from_confusion(totals["confusion"])
    for key, value in class_metrics.items():
        metrics[f"{prefix}{key}"] = value
    return metrics


def classification_metrics_from_confusion(confusion):
    confusion = confusion.to(dtype=torch.float64)
    tp = confusion.diag()
    pred_count = confusion.sum(dim=0).clamp_min(1.0)
    true_count = confusion.sum(dim=1).clamp_min(1.0)
    precision = tp / pred_count
    recall = tp / true_count
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    weights = true_count / true_count.sum().clamp_min(1.0)
    return {
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "weighted_f1": float((f1 * weights).sum().item()),
    }
