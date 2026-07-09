"""Per-event diagnostics for hit-origin classifier evaluation."""

import math

import torch
import torch.nn.functional as F


def _to_1d_cpu_tensor(value, dtype=None):
    if value is None:
        return None
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.reshape(-1)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _to_2d_cpu_tensor(value, dtype=None):
    if value is None:
        return None
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean_or_none(values):
    if values is None or values.numel() == 0:
        return None
    return _finite_float(values.to(dtype=torch.float64).mean().item())


def _quantile_or_none(values, quantile):
    if values is None or values.numel() == 0:
        return None
    return _finite_float(torch.quantile(values.to(dtype=torch.float64), float(quantile)).item())


def _optional_aligned_hit_vector(view, names, num_hits, dtype=torch.float32):
    for name in names:
        if name not in view:
            continue
        values = _to_1d_cpu_tensor(view[name], dtype=dtype)
        if values is not None and values.shape[0] == num_hits:
            return values
    return None


def prediction_metric_summary(
    true_class,
    pred_class,
    logits=None,
    energy_weights=None,
    high_confidence_threshold=0.8,
):
    """Return scalar per-event prediction diagnostics."""
    true_class = _to_1d_cpu_tensor(true_class, dtype=torch.long)
    pred_class = _to_1d_cpu_tensor(pred_class, dtype=torch.long)
    if true_class is None or pred_class is None or true_class.shape != pred_class.shape:
        raise ValueError("true_class and pred_class must be aligned one-dimensional vectors.")

    num_hits = int(true_class.numel())
    correct_mask = true_class == pred_class
    incorrect_mask = ~correct_mask
    correct_hits = int(correct_mask.sum().item())
    record = {
        "num_hits": num_hits,
        "correct_hits": correct_hits,
        "incorrect_hits": num_hits - correct_hits,
        "accuracy": None if num_hits == 0 else correct_hits / num_hits,
    }

    logits = _to_2d_cpu_tensor(logits, dtype=torch.float32)
    if logits is not None:
        if logits.ndim != 2 or logits.shape[0] != num_hits:
            raise ValueError(
                "logits must have shape [num_hits, num_classes], "
                f"got {tuple(logits.shape)} for {num_hits} hit(s)."
            )
        probabilities = F.softmax(logits, dim=1)
        confidence, _prob_pred_class = probabilities.max(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        if probabilities.shape[1] > 1:
            entropy = entropy / math.log(probabilities.shape[1])
            top2 = probabilities.topk(k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.ones_like(confidence)

        valid_true = (true_class >= 0) & (true_class < probabilities.shape[1])
        true_probability = torch.full_like(confidence, float("nan"))
        if bool(valid_true.any().item()):
            true_probability[valid_true] = probabilities[valid_true, true_class[valid_true]]

        record.update(
            {
                "mean_confidence": _mean_or_none(confidence),
                "mean_true_probability": _mean_or_none(true_probability[torch.isfinite(true_probability)]),
                "mean_margin": _mean_or_none(margin),
                "mean_entropy": _mean_or_none(entropy),
                "correct_mean_confidence": _mean_or_none(confidence[correct_mask]),
                "wrong_mean_confidence": _mean_or_none(confidence[incorrect_mask]),
                "high_confidence_error_fraction": (
                    None
                    if num_hits == 0
                    else float(((confidence >= high_confidence_threshold) & incorrect_mask).sum().item())
                    / num_hits
                ),
            }
        )

    weights = _to_1d_cpu_tensor(energy_weights, dtype=torch.float32)
    if weights is not None and weights.shape[0] == num_hits:
        weights = weights.clamp_min(0.0)
        total_weight = float(weights.sum().item())
        if total_weight > 0.0:
            correct_weight = float(weights[correct_mask].sum().item())
            incorrect_weight = total_weight - correct_weight
            record.update(
                {
                    "total_energy_weight": total_weight,
                    "correct_energy_weight": correct_weight,
                    "incorrect_energy_weight": incorrect_weight,
                    "energy_weighted_accuracy": correct_weight / total_weight,
                    "wrong_energy_fraction": incorrect_weight / total_weight,
                }
            )
        else:
            record.update(
                {
                    "total_energy_weight": total_weight,
                    "correct_energy_weight": None,
                    "incorrect_energy_weight": None,
                    "energy_weighted_accuracy": None,
                    "wrong_energy_fraction": None,
                }
            )

    return record


def _centroids_by_label(pos, labels, dims):
    valid = labels >= 0
    if not bool(valid.any().item()):
        return None, []
    present_labels = sorted({int(label) for label in labels[valid].tolist()})
    centroids = []
    for label in present_labels:
        mask = labels == label
        if bool(mask.any().item()):
            centroids.append(pos[mask][:, dims].mean(dim=0))
    if not centroids:
        return None, []
    return torch.stack(centroids, dim=0), present_labels


def _centroid_distance_stats(centroids, radius):
    if centroids is None or centroids.shape[0] < 2:
        return None, None, None
    pairwise = torch.pdist(centroids.to(dtype=torch.float64))
    min_distance = _finite_float(pairwise.min().item()) if pairwise.numel() else None
    mean_distance = _finite_float(pairwise.mean().item()) if pairwise.numel() else None
    within_radius = None
    if radius is not None and radius > 0:
        distances = torch.cdist(centroids.to(dtype=torch.float64), centroids.to(dtype=torch.float64))
        within_radius = int((distances <= float(radius)).sum(dim=1).max().item())
    return min_distance, mean_distance, within_radius


def _hit_centroid_margin_summary(pos_xy, labels):
    centroids, present_labels = _centroids_by_label(pos_xy, labels, dims=[0, 1])
    if centroids is None or centroids.shape[0] < 2:
        return {}
    label_to_idx = {label: idx for idx, label in enumerate(present_labels)}
    label_indices = torch.tensor(
        [label_to_idx.get(int(label), -1) for label in labels.tolist()],
        dtype=torch.long,
    )
    valid = label_indices >= 0
    if not bool(valid.any().item()):
        return {}

    distances = torch.cdist(pos_xy[valid].to(dtype=torch.float64), centroids.to(dtype=torch.float64))
    row = torch.arange(distances.shape[0])
    own_indices = label_indices[valid]
    own_distance = distances[row, own_indices]
    other_distances = distances.clone()
    other_distances[row, own_indices] = float("inf")
    nearest_other_distance = other_distances.min(dim=1).values
    finite_other = torch.isfinite(nearest_other_distance)
    if not bool(finite_other.any().item()):
        return {}
    margin = nearest_other_distance[finite_other] - own_distance[finite_other]
    return {
        "mean_hit_distance_to_origin_centroid_xy": _mean_or_none(own_distance),
        "p90_hit_distance_to_origin_centroid_xy": _quantile_or_none(own_distance, 0.9),
        "mean_hit_centroid_margin_xy": _mean_or_none(margin),
        "min_hit_centroid_margin_xy": _finite_float(margin.min().item()),
    }


def geometry_metric_summary(
    view,
    true_class,
    centroid_radius_mm=25.0,
    first_layer_tolerance_mm=1e-3,
):
    """Return event geometry and shower-overlap summaries from ECal positions."""
    true_class = _to_1d_cpu_tensor(true_class, dtype=torch.long)
    num_hits = 0 if true_class is None else int(true_class.numel())
    pos = _to_2d_cpu_tensor(view.get("ecal_pos"), dtype=torch.float32)
    if pos is None or pos.ndim != 2 or pos.shape != (num_hits, 3):
        return {}

    labels = _optional_aligned_hit_vector(
        view,
        names=("origin_id_y", "physical_y", "canonical_y", "y"),
        num_hits=num_hits,
        dtype=torch.long,
    )
    if labels is None:
        labels = true_class

    valid_labels = labels >= 0
    present_labels = sorted({int(label) for label in labels[valid_labels].tolist()})
    z_values = pos[:, 2]
    xy = pos[:, :2]

    summary = {
        "diagnostic_centroid_radius_mm": float(centroid_radius_mm),
        "num_truth_classes": len(present_labels),
        "num_ecal_layers": int(torch.unique(z_values).numel()),
        "ecal_z_min": _finite_float(z_values.min().item()) if num_hits else None,
        "ecal_z_max": _finite_float(z_values.max().item()) if num_hits else None,
        "ecal_z_span": _finite_float((z_values.max() - z_values.min()).item()) if num_hits else None,
    }

    centroids_3d, _labels_3d = _centroids_by_label(pos, labels, dims=[0, 1, 2])
    min_distance, mean_distance, within_radius = _centroid_distance_stats(
        centroids_3d,
        radius=centroid_radius_mm,
    )
    summary.update(
        {
            "min_origin_centroid_distance_3d": min_distance,
            "mean_origin_centroid_distance_3d": mean_distance,
            "max_origin_centroids_within_radius_3d": within_radius,
        }
    )

    centroids_xy, _labels_xy = _centroids_by_label(pos, labels, dims=[0, 1])
    min_distance, mean_distance, within_radius = _centroid_distance_stats(
        centroids_xy,
        radius=centroid_radius_mm,
    )
    summary.update(
        {
            "min_origin_centroid_distance_xy": min_distance,
            "mean_origin_centroid_distance_xy": mean_distance,
            "max_origin_centroids_within_radius_xy": within_radius,
        }
    )
    summary.update(_hit_centroid_margin_summary(xy, labels))

    if num_hits:
        first_z = z_values.min()
        first_layer_mask = torch.isclose(
            z_values,
            first_z,
            atol=float(first_layer_tolerance_mm),
            rtol=0.0,
        )
        first_labels = labels[first_layer_mask]
        first_pos = pos[first_layer_mask]
        first_valid_labels = first_labels >= 0
        first_present_labels = sorted(
            {int(label) for label in first_labels[first_valid_labels].tolist()}
        )
        first_centroids, _first_labels = _centroids_by_label(first_pos, first_labels, dims=[0, 1])
        min_distance, mean_distance, within_radius = _centroid_distance_stats(
            first_centroids,
            radius=centroid_radius_mm,
        )
        summary.update(
            {
                "first_layer_z": _finite_float(first_z.item()),
                "first_layer_num_hits": int(first_layer_mask.sum().item()),
                "first_layer_num_truth_classes": len(first_present_labels),
                "first_layer_min_origin_centroid_distance_xy": min_distance,
                "first_layer_mean_origin_centroid_distance_xy": mean_distance,
                "first_layer_max_origin_centroids_within_radius_xy": within_radius,
            }
        )

    return summary


def event_diagnostic_record(
    event_idx,
    split_position,
    view,
    true_class,
    pred_class,
    loss=None,
    logits=None,
    centroid_radius_mm=25.0,
):
    """Build one JSON/CSV-safe event diagnostic record."""
    true_class = _to_1d_cpu_tensor(true_class, dtype=torch.long)
    pred_class = _to_1d_cpu_tensor(pred_class, dtype=torch.long)
    energy_weights = _optional_aligned_hit_vector(
        view,
        names=("ecal_input_energy", "ecal_energy"),
        num_hits=int(true_class.numel()),
        dtype=torch.float32,
    )
    record = {
        "event_idx": int(event_idx),
        "split_position": int(split_position),
        **prediction_metric_summary(
            true_class=true_class,
            pred_class=pred_class,
            logits=logits,
            energy_weights=energy_weights,
        ),
        "loss": None if loss is None else _finite_float(loss.detach().cpu().item()),
    }
    record.update(
        geometry_metric_summary(
            view=view,
            true_class=true_class,
            centroid_radius_mm=centroid_radius_mm,
        )
    )
    return record


def select_representative_events(records, limit_per_group=3, metric="accuracy"):
    """Select worst, median, and best events by one scalar metric."""
    valid_records = []
    for record in records:
        value = record.get(metric)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid_records.append(record)
    if not valid_records:
        return {"metric": metric, "worst": [], "median": [], "best": []}

    limit = max(0, int(limit_per_group))
    median_value = sorted(float(record[metric]) for record in valid_records)[len(valid_records) // 2]

    def ascending_key(record):
        return (
            float(record[metric]),
            -int(record.get("incorrect_hits", 0)),
            int(record.get("event_idx", 0)),
        )

    def descending_key(record):
        return (
            -float(record[metric]),
            int(record.get("incorrect_hits", 0)),
            int(record.get("event_idx", 0)),
        )

    def median_key(record):
        return (
            abs(float(record[metric]) - median_value),
            int(record.get("incorrect_hits", 0)),
            int(record.get("event_idx", 0)),
        )

    return {
        "metric": metric,
        "median_value": median_value,
        "worst": sorted(valid_records, key=ascending_key)[:limit],
        "median": sorted(valid_records, key=median_key)[:limit],
        "best": sorted(valid_records, key=descending_key)[:limit],
    }
