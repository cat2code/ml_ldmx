"""Offline comparison helpers for event-level hit-classifier diagnostics."""

import csv
import json
import math
import re
from pathlib import Path

import numpy as np


PROFILE_METRICS = {
    "contributor_min_normalized_shower_separation_xy": (
        "any-contributor min normalized separation XY"
    ),
    "early_contributor_min_normalized_shower_separation_xy": (
        "first-three-layer any-contributor min normalized separation XY"
    ),
    "dominant_min_normalized_shower_separation_xy": (
        "dominant-only min normalized separation XY"
    ),
    "early_dominant_min_normalized_shower_separation_xy": (
        "first-three-layer dominant-only min normalized separation XY"
    ),
    "ambiguous_hit_fraction_xy": "geometrically ambiguous hit fraction",
    "num_hits": "ECal hits",
}
COMPARISON_TARGETS = {
    "accuracy": "event hit accuracy",
    "energy_weighted_accuracy": "event energy-weighted accuracy",
}
GEOMETRY_FIELDS = (
    "num_hits",
    "electron_count",
    "num_truth_classes",
    "min_origin_centroid_distance_xy",
    "first_layer_min_origin_centroid_distance_xy",
    "early_min_origin_centroid_distance_xy",
    "min_normalized_shower_separation_xy",
    "early_min_normalized_shower_separation_xy",
    "contributor_min_centroid_distance_xy",
    "contributor_min_normalized_shower_separation_xy",
    "early_contributor_min_normalized_shower_separation_xy",
    "first_layer_contributor_min_centroid_distance_xy",
    "dominant_min_centroid_distance_xy",
    "dominant_min_normalized_shower_separation_xy",
    "early_dominant_min_normalized_shower_separation_xy",
    "first_layer_dominant_min_centroid_distance_xy",
    "mean_shower_width_xy",
    "early_mean_shower_width_xy",
    "ambiguous_hit_fraction_xy",
    "energy_weighted_ambiguous_hit_fraction_xy",
    "mean_hit_centroid_margin_xy",
)


def label_slug(label):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(label)).strip("_").lower()
    return slug or "model"


def resolve_diagnostic_path(path, split="val", checkpoint="best"):
    """Resolve a run directory, inspection directory, or records file."""
    path = Path(path)
    if path.is_file():
        return path.resolve()
    filename_json = f"{split}_event_accuracy.json"
    filename_csv = f"{split}_event_accuracy.csv"
    candidates = [
        path / filename_json,
        path / filename_csv,
        path / "inspection" / checkpoint / split / filename_json,
        path / "inspection" / checkpoint / split / filename_csv,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find {split} event diagnostics under {path}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _parse_csv_value(value):
    if value == "":
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def load_event_records(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    elif path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            records = [
                {key: _parse_csv_value(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    else:
        raise ValueError(f"Expected a JSON or CSV diagnostics file, got {path}.")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected a list of event records in {path}.")
    return records


def event_identity(record):
    source_file = record.get("source_file")
    if source_file not in (None, ""):
        source_name = Path(str(source_file)).name
        source_entry = record.get("source_entry")
        source_label = record.get("source_label")
        electron_count = record.get("electron_count")
        return "source|{}|{}|{}|{}".format(
            "" if source_label is None else source_label,
            "" if electron_count is None else electron_count,
            source_name,
            "" if source_entry is None else source_entry,
        )
    event_idx = record.get("event_idx")
    if event_idx is None:
        raise ValueError("Event record has neither source_file nor event_idx identity metadata.")
    return f"event_idx|{event_idx}"


def _index_records(records, label):
    indexed = {}
    for record in records:
        key = event_identity(record)
        if key in indexed:
            raise ValueError(f"Run {label!r} contains duplicate event identity {key!r}.")
        indexed[key] = record
    return indexed


def _finite_number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _validate_matched_geometry(key, first_record, second_record):
    first_hits = first_record.get("num_hits")
    second_hits = second_record.get("num_hits")
    if first_hits is not None and second_hits is not None and int(first_hits) != int(second_hits):
        raise ValueError(
            f"Matched event {key!r} has different hit counts: {first_hits} versus {second_hits}."
        )
    for field in GEOMETRY_FIELDS:
        first_value = _finite_number(first_record.get(field))
        second_value = _finite_number(second_record.get(field))
        if first_value is None or second_value is None:
            continue
        if not math.isclose(first_value, second_value, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"Matched event {key!r} disagrees on geometry field {field!r}: "
                f"{first_value} versus {second_value}. Re-run both inspectors on the same data."
            )


def match_record_sets(first_label, first_records, second_label, second_records, allow_partial=False):
    first = _index_records(first_records, first_label)
    second = _index_records(second_records, second_label)
    first_keys = set(first)
    second_keys = set(second)
    common_keys = first_keys & second_keys
    first_only = sorted(first_keys - second_keys)
    second_only = sorted(second_keys - first_keys)
    if not common_keys:
        raise ValueError("The two diagnostic sets have no matching events.")
    if (first_only or second_only) and not allow_partial:
        raise ValueError(
            "Diagnostic event sets differ: "
            f"{len(first_only)} only in {first_label!r}, "
            f"{len(second_only)} only in {second_label!r}. "
            "Use --allow-partial-match only when this is intentional."
        )

    ordered_keys = sorted(
        common_keys,
        key=lambda key: (
            int(first[key].get("split_position", first[key].get("event_idx", 0))),
            key,
        ),
    )
    matches = []
    for key in ordered_keys:
        _validate_matched_geometry(key, first[key], second[key])
        matches.append(
            {
                "event_key": key,
                "records": {
                    first_label: first[key],
                    second_label: second[key],
                },
            }
        )
    return matches, {
        "matched_events": len(matches),
        "first_only_events": len(first_only),
        "second_only_events": len(second_only),
        "first_only_keys": first_only,
        "second_only_keys": second_only,
    }


def flatten_matches(matches, labels):
    first_label, second_label = labels
    slugs = {label: label_slug(label) for label in labels}
    if slugs[first_label] == slugs[second_label]:
        raise ValueError("Run labels must produce distinct file-safe names.")
    rows = []
    for match in matches:
        first = match["records"][first_label]
        second = match["records"][second_label]
        row = {
            "event_key": match["event_key"],
            "event_idx": first.get("event_idx"),
            "split_position": first.get("split_position"),
            "source_file": first.get("source_file"),
            "source_entry": first.get("source_entry"),
            "source_label": first.get("source_label"),
        }
        for field in GEOMETRY_FIELDS:
            row[field] = first.get(field)
        for label, record in ((first_label, first), (second_label, second)):
            slug = slugs[label]
            for field in (
                "accuracy",
                "energy_weighted_accuracy",
                "loss",
                "correct_hits",
                "incorrect_hits",
            ):
                row[f"{slug}_{field}"] = record.get(field)
        first_accuracy = _finite_number(first.get("accuracy"))
        second_accuracy = _finite_number(second.get("accuracy"))
        row[f"accuracy_delta_{slugs[first_label]}_minus_{slugs[second_label]}"] = (
            first_accuracy - second_accuracy
            if first_accuracy is not None and second_accuracy is not None
            else None
        )
        rows.append(row)
    return rows, slugs


def mean_confidence_interval(values, bootstrap_samples=400, seed=7, exact_limit=20000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, None, "unavailable"
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean, "single_event"
    if bootstrap_samples <= 0 or values.size > exact_limit:
        half_width = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
        return mean, mean - half_width, mean + half_width, "normal_approximation"

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((int(bootstrap_samples),), dtype=float)
    for sample_idx in range(int(bootstrap_samples)):
        indices = rng.integers(0, values.size, size=values.size)
        bootstrap_means[sample_idx] = values[indices].mean()
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high), "event_bootstrap"


def quantile_edges(values, num_bins):
    """Return unique equal-population bin edges for finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.unique(values).size < 2:
        return None
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, int(num_bins) + 1)))
    return edges if edges.size >= 2 else None


def build_binned_profiles(
    rows,
    labels,
    slugs,
    num_bins=8,
    bootstrap_samples=400,
    seed=7,
):
    output = []
    for metric in PROFILE_METRICS:
        x_values = [_finite_number(row.get(metric)) for row in rows]
        finite_x = [value for value in x_values if value is not None]
        edges = quantile_edges(finite_x, num_bins=num_bins)
        if edges is None:
            continue
        for bin_idx in range(edges.size - 1):
            low = float(edges[bin_idx])
            high = float(edges[bin_idx + 1])
            selected = []
            for row, x_value in zip(rows, x_values):
                if x_value is None:
                    continue
                in_bin = low <= x_value <= high if bin_idx == edges.size - 2 else low <= x_value < high
                if in_bin:
                    selected.append((row, x_value))
            if not selected:
                continue
            x_mean = float(np.mean([item[1] for item in selected]))
            for target in COMPARISON_TARGETS:
                for label_idx, label in enumerate(labels):
                    values = [
                        _finite_number(row.get(f"{slugs[label]}_{target}"))
                        for row, _x_value in selected
                    ]
                    values = [value for value in values if value is not None]
                    if not values:
                        continue
                    mean, ci_low, ci_high, ci_method = mean_confidence_interval(
                        values,
                        bootstrap_samples=bootstrap_samples,
                        seed=seed + 1000 * label_idx + 31 * bin_idx,
                    )
                    output.append(
                        {
                            "metric": metric,
                            "metric_label": PROFILE_METRICS[metric],
                            "target": target,
                            "target_label": COMPARISON_TARGETS[target],
                            "model": label,
                            "bin_index": bin_idx,
                            "bin_low": low,
                            "bin_high": high,
                            "x_mean": x_mean,
                            "event_count": len(values),
                            "mean": mean,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "ci_method": ci_method,
                        }
                    )
    return output


def build_multiplicity_profiles(rows, labels, slugs, bootstrap_samples=400, seed=7):
    def row_multiplicity(row):
        value = row.get("electron_count")
        if value is None:
            value = row.get("num_truth_classes")
        value = _finite_number(value)
        return None if value is None else int(value)

    multiplicities = sorted(
        {
            multiplicity
            for row in rows
            for multiplicity in [row_multiplicity(row)]
            if multiplicity is not None
        }
    )
    output = []
    for multiplicity in multiplicities:
        selected = [
            row
            for row in rows
            if row_multiplicity(row) == multiplicity
        ]
        for label_idx, label in enumerate(labels):
            values = [
                _finite_number(row.get(f"{slugs[label]}_accuracy"))
                for row in selected
            ]
            values = [value for value in values if value is not None]
            if not values:
                continue
            mean, ci_low, ci_high, ci_method = mean_confidence_interval(
                values,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1000 * label_idx + multiplicity,
            )
            output.append(
                {
                    "electron_count": multiplicity,
                    "model": label,
                    "event_count": len(values),
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_method": ci_method,
                }
            )
    return output


def comparison_summary(rows, labels, slugs, bootstrap_samples=400, seed=7):
    summary = {"num_matched_events": len(rows), "models": {}}
    for label_idx, label in enumerate(labels):
        slug = slugs[label]
        accuracies = [
            value
            for row in rows
            for value in [_finite_number(row.get(f"{slug}_accuracy"))]
            if value is not None
        ]
        energy_accuracies = [
            value
            for row in rows
            for value in [_finite_number(row.get(f"{slug}_energy_weighted_accuracy"))]
            if value is not None
        ]
        correct_hits = sum(int(row.get(f"{slug}_correct_hits") or 0) for row in rows)
        total_hits = sum(int(row.get("num_hits") or 0) for row in rows)
        mean, low, high, method = mean_confidence_interval(
            accuracies,
            bootstrap_samples=bootstrap_samples,
            seed=seed + label_idx,
        )
        summary["models"][label] = {
            "hit_accuracy": correct_hits / total_hits if total_hits else None,
            "mean_event_accuracy": mean,
            "mean_event_accuracy_ci_low": low,
            "mean_event_accuracy_ci_high": high,
            "mean_event_accuracy_ci_method": method,
            "mean_energy_weighted_accuracy": (
                float(np.mean(energy_accuracies)) if energy_accuracies else None
            ),
        }

    delta_field = f"accuracy_delta_{slugs[labels[0]]}_minus_{slugs[labels[1]]}"
    deltas = [
        value
        for row in rows
        for value in [_finite_number(row.get(delta_field))]
        if value is not None
    ]
    mean, low, high, method = mean_confidence_interval(
        deltas,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 100,
    )
    summary["paired_accuracy_delta"] = {
        "definition": f"{labels[0]} minus {labels[1]}",
        "mean": mean,
        "median": float(np.median(deltas)) if deltas else None,
        "ci_low": low,
        "ci_high": high,
        "ci_method": method,
    }
    return summary


def interesting_event_groups(
    rows,
    labels,
    slugs,
    low_accuracy=0.6,
    high_accuracy=0.9,
    difference_threshold=0.1,
    limit=20,
):
    first_label, second_label = labels
    first_field = f"{slugs[first_label]}_accuracy"
    second_field = f"{slugs[second_label]}_accuracy"
    delta_field = f"accuracy_delta_{slugs[first_label]}_minus_{slugs[second_label]}"

    def compact(row):
        fields = (
            "event_key",
            "event_idx",
            "source_file",
            "source_entry",
            "electron_count",
            "num_truth_classes",
            "num_hits",
            "contributor_min_normalized_shower_separation_xy",
            "early_contributor_min_normalized_shower_separation_xy",
            "dominant_min_normalized_shower_separation_xy",
            "early_dominant_min_normalized_shower_separation_xy",
            "ambiguous_hit_fraction_xy",
            first_field,
            second_field,
            delta_field,
        )
        return {field: row.get(field) for field in fields}

    valid = [
        row
        for row in rows
        if _finite_number(row.get(first_field)) is not None
        and _finite_number(row.get(second_field)) is not None
    ]
    both_fail = [
        row
        for row in valid
        if float(row[first_field]) <= low_accuracy and float(row[second_field]) <= low_accuracy
    ]
    both_good = [
        row
        for row in valid
        if float(row[first_field]) >= high_accuracy and float(row[second_field]) >= high_accuracy
    ]
    first_better = [row for row in valid if float(row[delta_field]) >= difference_threshold]
    second_better = [row for row in valid if float(row[delta_field]) <= -difference_threshold]

    both_fail.sort(key=lambda row: (float(row[first_field]) + float(row[second_field])) / 2.0)
    both_good.sort(
        key=lambda row: (float(row[first_field]) + float(row[second_field])) / 2.0,
        reverse=True,
    )
    first_better.sort(key=lambda row: float(row[delta_field]), reverse=True)
    second_better.sort(key=lambda row: float(row[delta_field]))
    return {
        "thresholds": {
            "low_accuracy": float(low_accuracy),
            "high_accuracy": float(high_accuracy),
            "difference_threshold": float(difference_threshold),
        },
        "category_counts": {
            "both_fail": len(both_fail),
            f"{slugs[first_label]}_better": len(first_better),
            f"{slugs[second_label]}_better": len(second_better),
            "both_good": len(both_good),
        },
        "both_fail": [compact(row) for row in both_fail[:limit]],
        f"{slugs[first_label]}_better": [compact(row) for row in first_better[:limit]],
        f"{slugs[second_label]}_better": [compact(row) for row in second_better[:limit]],
        "both_good": [compact(row) for row in both_good[:limit]],
    }
