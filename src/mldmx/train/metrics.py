import torch


def confusion_matrix_from_class_indices(true_class, pred_class, num_classes):
    """Return a true-row/pred-column confusion matrix using vectorized bincount."""
    true_class = true_class.detach().reshape(-1).to(dtype=torch.long)
    pred_class = pred_class.detach().reshape(-1).to(device=true_class.device, dtype=torch.long)
    if true_class.shape != pred_class.shape:
        raise ValueError(
            "true_class and pred_class must have the same flattened shape, "
            f"got {tuple(true_class.shape)} and {tuple(pred_class.shape)}."
        )
    if true_class.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long, device=true_class.device)

    linear_indices = true_class * int(num_classes) + pred_class
    return torch.bincount(
        linear_indices,
        minlength=int(num_classes) * int(num_classes),
    ).reshape(int(num_classes), int(num_classes))


def confusion_matrix_from_labels(y_true, y_pred, labels):
    """
    Return a true-row/pred-column confusion matrix for arbitrary integer labels.

    Labels not present in ``labels`` are ignored, matching the previous plotting
    helper behavior.
    """
    labels = torch.as_tensor(labels, dtype=torch.long)
    true_values = torch.as_tensor(y_true, dtype=torch.long).reshape(-1)
    pred_values = torch.as_tensor(y_pred, dtype=torch.long).reshape(-1)
    if true_values.shape != pred_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same flattened shape, "
            f"got {tuple(true_values.shape)} and {tuple(pred_values.shape)}."
        )
    if labels.numel() == 0:
        return torch.zeros((0, 0), dtype=torch.long)

    labels = labels.to(device=true_values.device)
    pred_values = pred_values.to(device=true_values.device)
    true_matches = true_values[:, None] == labels[None, :]
    pred_matches = pred_values[:, None] == labels[None, :]
    valid = true_matches.any(dim=1) & pred_matches.any(dim=1)
    if not bool(valid.any().item()):
        return torch.zeros((labels.numel(), labels.numel()), dtype=torch.long, device=true_values.device)

    true_indices = true_matches[valid].to(dtype=torch.long).argmax(dim=1)
    pred_indices = pred_matches[valid].to(dtype=torch.long).argmax(dim=1)
    return confusion_matrix_from_class_indices(true_indices, pred_indices, labels.numel())


def confusion_components_from_confusion(confusion):
    """Return per-class TP, FP, TN, and FN vectors from a confusion matrix."""
    confusion = confusion.to(dtype=torch.float64)
    tp = confusion.diag()
    predicted = confusion.sum(dim=0)
    actual = confusion.sum(dim=1)
    fp = predicted - tp
    fn = actual - tp
    tn = confusion.sum() - tp - fp - fn
    return tp, fp, tn, fn


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
    loss_values = torch.stack(
        [
            losses["total_loss"],
            losses["origin_loss"],
            losses["fraction_loss"],
            losses["fraction_mse"],
            losses["fraction_mae"],
        ]
    ).detach().cpu()
    totals["loss_sum"] += float(loss_values[0].item()) * num_hits
    totals["origin_loss_sum"] += float(loss_values[1].item()) * num_hits
    totals["fraction_loss_sum"] += float(loss_values[2].item()) * num_hits
    totals["fraction_mse_sum"] += float(loss_values[3].item()) * num_hits
    totals["fraction_mae_sum"] += float(loss_values[4].item()) * num_hits

    confusion = confusion_matrix_from_class_indices(
        losses["true_class"],
        losses["pred_class"],
        totals["confusion"].shape[0],
    ).cpu()
    totals["correct"] += int(confusion.diag().sum().item())
    totals["hits"] += num_hits
    totals["confusion"] += confusion


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
    tp, fp, _tn, fn = confusion_components_from_confusion(confusion)
    pred_count = (tp + fp).clamp_min(1.0)
    true_count = (tp + fn).clamp_min(1.0)
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
