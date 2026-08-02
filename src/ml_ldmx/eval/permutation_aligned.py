"""Reusable metrics for permutation-aligned hit-grouping evaluation."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from ml_ldmx.eval.event_diagnostics import optimal_label_permutation_summary
from ml_ldmx.eval.run_comparison import mean_confidence_interval, quantile_edges


@dataclass
class AlignedEventPrediction:
    """Per-hit arrays retained from one event after whole-event label alignment."""

    event_idx: int
    split_position: int
    true_class: np.ndarray
    predicted_class: np.ndarray
    aligned_predicted_class: np.ndarray
    optimal_mapping: np.ndarray
    confidence: np.ndarray
    normalized_entropy: np.ndarray
    probability_margin: np.ndarray
    raw_energy: np.ndarray
    position: np.ndarray
    origin_fraction: np.ndarray
    electron_count: int | None = None
    layer: np.ndarray | None = None

    @property
    def aligned_correct(self):
        return self.true_class == self.aligned_predicted_class

    @property
    def ordinary_correct(self):
        return self.true_class == self.predicted_class


def _cpu_tensor(value, dtype=None):
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def apply_prediction_label_mapping(predicted_class, mapping):
    """Map predicted class indices into truth-class indices."""
    predicted_class = _cpu_tensor(predicted_class, dtype=torch.long).reshape(-1)
    mapping = _cpu_tensor(mapping, dtype=torch.long).reshape(-1)
    if predicted_class.numel() and (
        int(predicted_class.min().item()) < 0
        or int(predicted_class.max().item()) >= mapping.numel()
    ):
        raise ValueError("Predicted class lies outside the supplied label mapping.")
    return mapping[predicted_class]


def aligned_event_prediction(prediction, num_classes):
    """Convert one evaluator prediction into CPU arrays with one event-level mapping."""
    true_class = _cpu_tensor(prediction["true_class"], dtype=torch.long).reshape(-1)
    predicted_class = _cpu_tensor(prediction["pred_class"], dtype=torch.long).reshape(-1)
    logits = _cpu_tensor(prediction["logits"], dtype=torch.float32)
    if true_class.shape != predicted_class.shape:
        raise ValueError("Truth and prediction arrays must have the same shape.")
    if logits.ndim != 2 or logits.shape != (true_class.numel(), int(num_classes)):
        raise ValueError(
            "Expected logits with shape "
            f"[{true_class.numel()}, {int(num_classes)}], got {tuple(logits.shape)}."
        )

    view = prediction["view"]
    raw_energy = _cpu_tensor(view.get("ecal_raw_energy"), dtype=torch.float32).reshape(-1)
    position = _cpu_tensor(view.get("ecal_pos"), dtype=torch.float32)
    origin_fraction_value = view.get(
        "origin_id_fraction_target",
        view.get("fraction_target"),
    )
    origin_fraction = _cpu_tensor(origin_fraction_value, dtype=torch.float32)
    num_hits = true_class.numel()
    if raw_energy.shape != (num_hits,):
        raise ValueError("ecal_raw_energy must align with supervised ECal hits.")
    if position.shape != (num_hits, 3):
        raise ValueError("ecal_pos must have shape [num_hits, 3].")
    if origin_fraction.ndim != 2 or origin_fraction.shape[0] != num_hits:
        raise ValueError("Origin-fraction targets must align with supervised ECal hits.")

    permutation = optimal_label_permutation_summary(
        true_class=true_class,
        pred_class=predicted_class,
        num_classes=num_classes,
    )
    mapping = torch.as_tensor(
        permutation["optimal_prediction_label_mapping"],
        dtype=torch.long,
    )
    aligned_prediction = apply_prediction_label_mapping(predicted_class, mapping)

    probabilities = F.softmax(logits, dim=1)
    confidence = probabilities.max(dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    if int(num_classes) > 1:
        entropy = entropy / math.log(int(num_classes))
        top_two = probabilities.topk(k=2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
    else:
        entropy = torch.zeros_like(confidence)
        margin = torch.ones_like(confidence)

    electron_count = view.get("electron_count")
    if electron_count is not None:
        electron_count = int(_cpu_tensor(electron_count).reshape(-1)[0].item())

    return AlignedEventPrediction(
        event_idx=int(prediction["event_idx"]),
        split_position=int(prediction["split_position"]),
        true_class=true_class.numpy(),
        predicted_class=predicted_class.numpy(),
        aligned_predicted_class=aligned_prediction.numpy(),
        optimal_mapping=mapping.numpy(),
        confidence=confidence.numpy(),
        normalized_entropy=entropy.numpy(),
        probability_margin=margin.numpy(),
        raw_energy=raw_energy.numpy(),
        position=position.numpy(),
        origin_fraction=origin_fraction.numpy(),
        electron_count=electron_count,
    )


def global_layer_z(events, expected_layers=None, decimals=3):
    """Return the globally ordered detector-layer z coordinates."""
    z_parts = [
        np.asarray(event.position[:, 2], dtype=np.float64)
        for event in events
        if event.position.size
    ]
    if not z_parts:
        raise ValueError("Cannot construct an ECal layer lookup from no hit positions.")
    layer_z = np.unique(np.round(np.concatenate(z_parts), decimals=int(decimals)))
    layer_z.sort()
    if expected_layers is not None and layer_z.size != int(expected_layers):
        raise ValueError(
            f"Expected {int(expected_layers)} global ECal layers, found {layer_z.size}: "
            f"{layer_z.tolist()}."
        )
    return layer_z


def assign_global_layers(events, layer_z, tolerance_mm=0.01):
    """Attach one-based global detector-layer indices to every retained hit."""
    layer_z = np.asarray(layer_z, dtype=np.float64).reshape(-1)
    if layer_z.size == 0 or np.any(np.diff(layer_z) <= 0):
        raise ValueError("layer_z must be a non-empty, strictly increasing vector.")
    for event in events:
        z = np.asarray(event.position[:, 2], dtype=np.float64).reshape(-1)
        distances = np.abs(z[:, None] - layer_z[None, :])
        nearest = distances.argmin(axis=1)
        residual = distances[np.arange(z.size), nearest]
        if residual.size and float(residual.max()) > float(tolerance_mm):
            raise ValueError(
                f"Event {event.event_idx} contains an ECal z position farther than "
                f"{tolerance_mm} mm from the global layer lookup."
            )
        event.layer = nearest.astype(np.int16) + 1
    return events


def reconstructed_energy_origin_weights(raw_energy, origin_fraction):
    """Build nonnegative per-origin hit weights from reconstructed hit energy."""
    energy = np.asarray(raw_energy, dtype=np.float64).reshape(-1)
    fractions = np.asarray(origin_fraction, dtype=np.float64)
    if fractions.ndim != 2 or fractions.shape[0] != energy.size:
        raise ValueError("origin_fraction must have shape [num_hits, num_origins].")
    energy = np.where(np.isfinite(energy), np.clip(energy, 0.0, None), 0.0)
    fractions = np.where(np.isfinite(fractions), np.clip(fractions, 0.0, None), 0.0)
    return energy[:, None] * fractions


def energy_weighted_shower_geometry(
    position_xy,
    raw_energy,
    origin_fraction,
    moliere_radius_mm=25.0,
):
    """Return truth-origin shower moments weighted by reconstructed energy."""
    position_xy = np.asarray(position_xy, dtype=np.float64)
    if position_xy.ndim != 2 or position_xy.shape[1] != 2:
        raise ValueError("position_xy must have shape [num_hits, 2].")
    weights = reconstructed_energy_origin_weights(raw_energy, origin_fraction)
    if weights.shape[0] != position_xy.shape[0]:
        raise ValueError("Position and energy-weight arrays must contain the same hits.")

    total_weights = weights.sum(axis=0)
    active = np.isfinite(total_weights) & (total_weights > 0.0)
    weights = weights[:, active]
    total_weights = total_weights[active]
    if total_weights.size < 2:
        return {
            "num_showers": int(total_weights.size),
            "min_centroid_distance_mm": None,
            "mean_centroid_distance_mm": None,
            "min_centroid_distance_moliere": None,
            "mean_centroid_distance_moliere": None,
            "min_width_normalized_separation": None,
            "mean_width_normalized_separation": None,
            "mean_shower_width_mm": None,
        }

    centroids = (weights.T @ position_xy) / total_weights[:, None]
    displacement = position_xy[:, None, :] - centroids[None, :, :]
    squared_radius = np.sum(displacement * displacement, axis=2)
    widths = np.sqrt(np.sum(weights * squared_radius, axis=0) / total_weights)

    distances = []
    normalized_separations = []
    for first in range(centroids.shape[0] - 1):
        for second in range(first + 1, centroids.shape[0]):
            distance = float(np.linalg.norm(centroids[first] - centroids[second]))
            distances.append(distance)
            combined_width = math.sqrt(widths[first] ** 2 + widths[second] ** 2)
            if combined_width > 1e-12:
                normalized_separations.append(distance / combined_width)

    distances = np.asarray(distances, dtype=np.float64)
    normalized = np.asarray(normalized_separations, dtype=np.float64)
    radius = float(moliere_radius_mm)
    if radius <= 0.0:
        raise ValueError("moliere_radius_mm must be positive.")
    return {
        "num_showers": int(total_weights.size),
        "min_centroid_distance_mm": float(distances.min()),
        "mean_centroid_distance_mm": float(distances.mean()),
        "min_centroid_distance_moliere": float(distances.min() / radius),
        "mean_centroid_distance_moliere": float(distances.mean() / radius),
        "min_width_normalized_separation": (
            None if normalized.size == 0 else float(normalized.min())
        ),
        "mean_width_normalized_separation": (
            None if normalized.size == 0 else float(normalized.mean())
        ),
        "mean_shower_width_mm": float(widths.mean()),
    }


def aligned_event_metrics(event, moliere_radius_mm=25.0, early_layers=3):
    """Summarize one aligned event while keeping the full-event mapping fixed."""
    if event.layer is None:
        raise ValueError("Global ECal layers must be assigned before event summarization.")
    correct = event.aligned_correct
    ordinary_correct = event.ordinary_correct
    num_hits = int(correct.size)
    energy = np.clip(np.asarray(event.raw_energy, dtype=np.float64), 0.0, None)
    total_energy = float(energy.sum())

    full_geometry = energy_weighted_shower_geometry(
        event.position[:, :2],
        event.raw_energy,
        event.origin_fraction,
        moliere_radius_mm=moliere_radius_mm,
    )
    early_mask = event.layer <= int(early_layers)
    early_geometry = energy_weighted_shower_geometry(
        event.position[early_mask, :2],
        event.raw_energy[early_mask],
        event.origin_fraction[early_mask],
        moliere_radius_mm=moliere_radius_mm,
    )

    record = {
        "event_idx": event.event_idx,
        "split_position": event.split_position,
        "electron_count": event.electron_count,
        "num_hits": num_hits,
        "ordinary_correct_hits": int(ordinary_correct.sum()),
        "ordinary_event_accuracy": (
            None if num_hits == 0 else float(ordinary_correct.mean())
        ),
        "aligned_correct_hits": int(correct.sum()),
        "aligned_event_accuracy": None if num_hits == 0 else float(correct.mean()),
        "raw_reconstructed_energy": total_energy,
        "aligned_energy_weighted_accuracy": (
            None
            if total_energy <= 0.0
            else float(energy[correct].sum() / total_energy)
        ),
        "optimal_prediction_label_mapping": event.optimal_mapping.tolist(),
        "mapping_is_identity": bool(
            np.array_equal(event.optimal_mapping, np.arange(event.optimal_mapping.size))
        ),
        "mean_confidence": float(np.mean(event.confidence)),
        "p10_confidence": float(np.quantile(event.confidence, 0.1)),
        "min_confidence": float(np.min(event.confidence)),
        "confidence_standard_deviation": float(np.std(event.confidence)),
        "fraction_confidence_below_0p8": float(np.mean(event.confidence < 0.8)),
        "mean_normalized_entropy": float(np.mean(event.normalized_entropy)),
        "mean_probability_margin": float(np.mean(event.probability_margin)),
    }
    for prefix, geometry in (
        ("energy_weighted", full_geometry),
        (f"first_{int(early_layers)}_layers_energy_weighted", early_geometry),
    ):
        for key, value in geometry.items():
            record[f"{prefix}_{key}"] = value
    return record


def event_layer_metrics(event):
    """Return one row for each physical ECal layer represented in an event."""
    if event.layer is None:
        raise ValueError("Global ECal layers must be assigned before layer summarization.")
    rows = []
    correct = event.aligned_correct
    energy = np.clip(np.asarray(event.raw_energy, dtype=np.float64), 0.0, None)
    for layer in np.unique(event.layer):
        mask = event.layer == layer
        layer_energy = float(energy[mask].sum())
        rows.append(
            {
                "event_idx": event.event_idx,
                "split_position": event.split_position,
                "electron_count": event.electron_count,
                "layer": int(layer),
                "num_hits": int(mask.sum()),
                "aligned_correct_hits": int(correct[mask].sum()),
                "aligned_event_layer_accuracy": float(correct[mask].mean()),
                "raw_reconstructed_energy": layer_energy,
                "aligned_event_layer_energy_weighted_accuracy": (
                    None
                    if layer_energy <= 0.0
                    else float(energy[mask & correct].sum() / layer_energy)
                ),
                "mean_confidence": float(event.confidence[mask].mean()),
            }
        )
    return rows


def layer_profiles(layer_rows, bootstrap_samples=500, seed=7):
    """Aggregate event-balanced layer metrics with event-bootstrap intervals."""
    output = []
    metric_names = (
        "aligned_event_layer_accuracy",
        "aligned_event_layer_energy_weighted_accuracy",
        "mean_confidence",
    )
    layers = sorted({int(row["layer"]) for row in layer_rows})
    for layer in layers:
        selected = [row for row in layer_rows if int(row["layer"]) == layer]
        profile = {
            "layer": layer,
            "num_events": len(selected),
            "num_hits": sum(int(row["num_hits"]) for row in selected),
            "median_aligned_event_layer_accuracy": float(
                np.median(
                    [
                        float(row["aligned_event_layer_accuracy"])
                        for row in selected
                    ]
                )
            ),
        }
        for metric_index, metric in enumerate(metric_names):
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ]
            mean, low, high, method = mean_confidence_interval(
                values,
                bootstrap_samples=bootstrap_samples,
                seed=int(seed) + 101 * layer + 10007 * metric_index,
            )
            profile[f"{metric}_mean"] = mean
            profile[f"{metric}_ci_low"] = low
            profile[f"{metric}_ci_high"] = high
            profile[f"{metric}_ci_method"] = method
        output.append(profile)
    return output


def binned_event_profile(
    event_records,
    x_key,
    y_key="aligned_event_accuracy",
    num_bins=10,
    bootstrap_samples=500,
    seed=7,
):
    """Build an equal-population event profile for a scalar diagnostic."""
    pairs = []
    for row in event_records:
        x_value = row.get(x_key)
        y_value = row.get(y_key)
        if x_value is None or y_value is None:
            continue
        x_value = float(x_value)
        y_value = float(y_value)
        if np.isfinite(x_value) and np.isfinite(y_value):
            pairs.append((x_value, y_value))
    if not pairs:
        return []
    edges = quantile_edges([pair[0] for pair in pairs], num_bins=num_bins)
    if edges is None:
        return []

    output = []
    for bin_index in range(edges.size - 1):
        low = float(edges[bin_index])
        high = float(edges[bin_index + 1])
        selected = [
            pair
            for pair in pairs
            if (
                low <= pair[0] <= high
                if bin_index == edges.size - 2
                else low <= pair[0] < high
            )
        ]
        if not selected:
            continue
        y_values = [pair[1] for pair in selected]
        mean, ci_low, ci_high, ci_method = mean_confidence_interval(
            y_values,
            bootstrap_samples=bootstrap_samples,
            seed=int(seed) + 131 * bin_index,
        )
        output.append(
            {
                "x_key": x_key,
                "y_key": y_key,
                "bin": bin_index + 1,
                "x_low": low,
                "x_high": high,
                "x_mean": float(np.mean([pair[0] for pair in selected])),
                "num_events": len(selected),
                "y_mean": mean,
                "y_ci_low": ci_low,
                "y_ci_high": ci_high,
                "ci_method": ci_method,
            }
        )
    return output


def confusion_counts(events, num_classes, layer_min=None, layer_max=None):
    """Pool hit-count confusion entries after each event's fixed label alignment."""
    matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for event in events:
        mask = np.ones(event.true_class.shape, dtype=bool)
        if layer_min is not None or layer_max is not None:
            if event.layer is None:
                raise ValueError("Global ECal layers are required for depth confusion.")
            if layer_min is not None:
                mask &= event.layer >= int(layer_min)
            if layer_max is not None:
                mask &= event.layer <= int(layer_max)
        np.add.at(
            matrix,
            (event.true_class[mask], event.aligned_predicted_class[mask]),
            1,
        )
    return matrix


def calibration_bins(events, num_bins=10):
    """Pool aligned hit correctness in equal-width max-confidence bins."""
    confidence = np.concatenate([event.confidence for event in events])
    correct = np.concatenate([event.aligned_correct for event in events]).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    bin_index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, num_bins - 1)
    rows = []
    total = confidence.size
    ece = 0.0
    for index in range(int(num_bins)):
        mask = bin_index == index
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += count / total * abs(accuracy - mean_confidence)
        else:
            mean_confidence = None
            accuracy = None
        rows.append(
            {
                "bin": index + 1,
                "confidence_low": float(edges[index]),
                "confidence_high": float(edges[index + 1]),
                "num_hits": count,
                "mean_confidence": mean_confidence,
                "aligned_hit_accuracy": accuracy,
            }
        )
    return rows, float(ece)


def accuracy_coverage_profiles(events, points=20):
    """Return hit- and event-selection profiles without recomputing label mappings."""
    coverages = np.linspace(0.05, 1.0, int(points))
    hit_confidence = np.concatenate([event.confidence for event in events])
    hit_correct = np.concatenate([event.aligned_correct for event in events])
    hit_energy = np.concatenate([event.raw_energy for event in events]).astype(np.float64)
    hit_order = np.argsort(-hit_confidence, kind="stable")
    total_hit_energy = float(np.clip(hit_energy, 0.0, None).sum())

    hit_rows = []
    for coverage in coverages:
        retained = max(1, int(math.ceil(coverage * hit_order.size)))
        selected = hit_order[:retained]
        retained_energy = float(np.clip(hit_energy[selected], 0.0, None).sum())
        hit_rows.append(
            {
                "requested_hit_coverage": float(coverage),
                "retained_hits": retained,
                "hit_coverage": retained / hit_order.size,
                "minimum_retained_confidence": float(hit_confidence[selected].min()),
                "aligned_hit_accuracy": float(hit_correct[selected].mean()),
                "energy_coverage": (
                    None if total_hit_energy <= 0.0 else retained_energy / total_hit_energy
                ),
            }
        )

    event_confidence = np.asarray(
        [float(np.mean(event.confidence)) for event in events],
        dtype=np.float64,
    )
    event_accuracy = np.asarray(
        [float(np.mean(event.aligned_correct)) for event in events],
        dtype=np.float64,
    )
    event_order = np.argsort(-event_confidence, kind="stable")
    event_rows = []
    for coverage in coverages:
        retained = max(1, int(math.ceil(coverage * event_order.size)))
        selected = event_order[:retained]
        event_rows.append(
            {
                "requested_event_coverage": float(coverage),
                "retained_events": retained,
                "event_coverage": retained / event_order.size,
                "minimum_retained_mean_confidence": float(
                    event_confidence[selected].min()
                ),
                "macro_aligned_event_accuracy": float(event_accuracy[selected].mean()),
            }
        )
    return hit_rows, event_rows
