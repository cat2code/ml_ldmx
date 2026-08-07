"""Generate model-independent ECal geometry plots from the 20k tensor samples."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.datasets.ecal_tpad_shards import ShardedECalTpadDataset
from ml_ldmx.io.artifacts import save_json


DEFAULT_SAMPLES = (
    (
        "2e",
        PROJECT_ROOT / "data/ldmx_overlay_events_700k_shards_log1p/2e/events",
    ),
    (
        "3e",
        PROJECT_ROOT / "data/ldmx_overlay_events_700k_shards_log1p/3e/events",
    ),
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/plots_development/006_physical_geometry_20k"
COLORS = ("#2563eb", "#d97706", "#0f766e", "#7c3aed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="append",
        nargs=2,
        metavar=("LABEL", "CACHE_DIR"),
        help="Sample label and sharded tensor cache. Repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-events",
        type=int,
        default=20_000,
        help="Number of deterministic cache-prefix events per sample (default: 20000).",
    )
    parser.add_argument("--progress-every", type=int, default=2_000)
    return parser.parse_args()


def _numpy(value, dtype=np.float64):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _scalar_int(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()
    return int(value)


def _origin_geometry(event):
    """Return energy-fraction-weighted origin centroids and transverse widths."""
    position = _numpy(event["ecal_pos"])
    raw_energy = np.clip(_numpy(event["ecal_raw_energy"]).reshape(-1), 0.0, None)
    fraction_value = event.get(
        "origin_id_fraction_target",
        event.get("fraction_target"),
    )
    if fraction_value is None:
        raise KeyError("Event has no physical-origin fraction target.")
    fractions = _numpy(fraction_value)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("ecal_pos must have shape [num_hits, 3].")
    if fractions.ndim != 2 or fractions.shape[0] != position.shape[0]:
        raise ValueError("Origin fractions must align with ECal hits.")
    fractions = np.where(
        np.isfinite(fractions),
        np.clip(fractions, 0.0, None),
        0.0,
    )
    weights = raw_energy[:, None] * fractions
    total_weight = weights.sum(axis=0)
    active = np.isfinite(total_weight) & (total_weight > 0.0)
    weights = weights[:, active]
    total_weight = total_weight[active]
    if total_weight.size == 0:
        return {
            "position": position,
            "raw_energy": raw_energy,
            "origin_labels": [],
            "centroids": np.empty((0, 2), dtype=float),
            "widths": np.empty((0, 3), dtype=float),
            "origin_energy": np.empty((0,), dtype=float),
        }

    label_order = event.get(
        "origin_id_fraction_label_order",
        event.get("target_label_order", range(1, fractions.shape[1] + 1)),
    )
    label_order = list(label_order)
    if len(label_order) != fractions.shape[1]:
        raise ValueError("Fraction-label order does not match fraction columns.")
    origin_labels = [int(label_order[index]) for index in np.flatnonzero(active)]

    centroids = (weights.T @ position[:, :2]) / total_weight[:, None]
    displacement = position[:, None, :2] - centroids[None, :, :]
    variances = (
        np.sum(weights[:, :, None] * displacement * displacement, axis=0)
        / total_weight[:, None]
    )
    variances = np.clip(variances, 0.0, None)
    widths_xy = np.sqrt(variances)
    radial_width = np.sqrt(np.sum(variances, axis=1))
    widths = np.column_stack([widths_xy, radial_width])
    return {
        "position": position,
        "raw_energy": raw_energy,
        "origin_labels": origin_labels,
        "centroids": centroids,
        "widths": widths,
        "origin_energy": total_weight,
    }


def _pairwise_axis_separations(centroids):
    centroids = np.asarray(centroids, dtype=float)
    if centroids.shape[0] < 2:
        return np.empty((0, 2), dtype=float)
    return np.asarray(
        [
            np.abs(centroids[first] - centroids[second])
            for first in range(centroids.shape[0] - 1)
            for second in range(first + 1, centroids.shape[0])
        ],
        dtype=float,
    )


def _scan_sample(label, cache_dir, max_events, progress_every):
    dataset = ShardedECalTpadDataset(
        cache_dir,
        max_events=max_events,
        shard_cache_size=1,
    )
    event_rows = []
    shower_rows = []
    layer_sums = {}
    num_origin_count_mismatches = 0
    for event_idx in range(len(dataset)):
        event = dataset[event_idx]
        geometry = _origin_geometry(event)
        position = geometry["position"]
        raw_energy = geometry["raw_energy"]
        centroids = geometry["centroids"]
        widths = geometry["widths"]
        origin_labels = geometry["origin_labels"]
        electron_count = _scalar_int(event.get("electron_count", len(origin_labels)))
        if len(origin_labels) != electron_count:
            num_origin_count_mismatches += 1

        separations = _pairwise_axis_separations(centroids)
        event_rows.append(
            {
                "sample": label,
                "event_idx": event_idx,
                "source_file": event.get("source_file", ""),
                "source_entry": int(event.get("source_entry", -1)),
                "electron_count": electron_count,
                "num_active_truth_origins": len(origin_labels),
                "num_origin_pairs": int(separations.shape[0]),
                "min_abs_delta_mu_x_mm": (
                    None if separations.size == 0 else float(separations[:, 0].min())
                ),
                "min_abs_delta_mu_y_mm": (
                    None if separations.size == 0 else float(separations[:, 1].min())
                ),
                "mean_abs_delta_mu_x_mm": (
                    None if separations.size == 0 else float(separations[:, 0].mean())
                ),
                "mean_abs_delta_mu_y_mm": (
                    None if separations.size == 0 else float(separations[:, 1].mean())
                ),
            }
        )
        for origin_index, origin_label in enumerate(origin_labels):
            shower_rows.append(
                {
                    "sample": label,
                    "event_idx": event_idx,
                    "origin_label": origin_label,
                    "centroid_x_mm": float(centroids[origin_index, 0]),
                    "centroid_y_mm": float(centroids[origin_index, 1]),
                    "sigma_x_mm": float(widths[origin_index, 0]),
                    "sigma_y_mm": float(widths[origin_index, 1]),
                    "sigma_r_mm": float(widths[origin_index, 2]),
                    "reconstructed_origin_energy_mev": float(
                        geometry["origin_energy"][origin_index]
                    ),
                }
            )

        z_values, inverse = np.unique(np.round(position[:, 2], 3), return_inverse=True)
        hit_counts = np.bincount(inverse, minlength=z_values.size).astype(float)
        energy_sums = np.bincount(
            inverse,
            weights=raw_energy,
            minlength=z_values.size,
        ).astype(float)
        for z_value, hit_count, energy_sum in zip(z_values, hit_counts, energy_sums):
            accumulator = layer_sums.setdefault(
                float(z_value),
                {"hit_sum": 0.0, "hit_sum_squares": 0.0, "energy_sum": 0.0, "energy_sum_squares": 0.0},
            )
            accumulator["hit_sum"] += float(hit_count)
            accumulator["hit_sum_squares"] += float(hit_count * hit_count)
            accumulator["energy_sum"] += float(energy_sum)
            accumulator["energy_sum_squares"] += float(energy_sum * energy_sum)

        if progress_every > 0 and (event_idx + 1) % progress_every == 0:
            print(f"{label}: scanned {event_idx + 1:,}/{len(dataset):,} events", flush=True)

    layer_rows = _finalize_layer_rows(label, layer_sums, len(dataset))
    return {
        "label": label,
        "cache_dir": str(Path(cache_dir).resolve()),
        "num_events": len(dataset),
        "event_rows": event_rows,
        "shower_rows": shower_rows,
        "layer_rows": layer_rows,
        "num_origin_count_mismatches": num_origin_count_mismatches,
    }


def _mean_ci_from_sums(total, total_squares, count):
    mean = total / count
    if count <= 1:
        return mean, mean, mean
    variance = max(0.0, (total_squares - total * total / count) / (count - 1))
    half_width = 1.96 * math.sqrt(variance / count)
    return mean, mean - half_width, mean + half_width


def _finalize_layer_rows(label, layer_sums, num_events):
    rows = []
    for layer_index, z_value in enumerate(sorted(layer_sums), start=1):
        values = layer_sums[z_value]
        hit_mean, hit_low, hit_high = _mean_ci_from_sums(
            values["hit_sum"],
            values["hit_sum_squares"],
            num_events,
        )
        energy_mean, energy_low, energy_high = _mean_ci_from_sums(
            values["energy_sum"],
            values["energy_sum_squares"],
            num_events,
        )
        rows.append(
            {
                "sample": label,
                "layer": layer_index,
                "z_mm": z_value,
                "mean_hits_per_event": hit_mean,
                "hit_mean_ci95_low": hit_low,
                "hit_mean_ci95_high": hit_high,
                "mean_reconstructed_energy_per_event_mev": energy_mean,
                "energy_mean_ci95_low": energy_low,
                "energy_mean_ci95_high": energy_high,
            }
        )
    return rows


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_figure(fig, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _finite_values(rows, key):
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _nice_upper(values, quantile=0.995):
    upper = float(np.quantile(np.asarray(values, dtype=float), quantile))
    if upper <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(upper))
    step = magnitude / 2.0
    return max(step, math.ceil(upper / step) * step)


def _distribution_edges(analyses, row_key, metric_key, bins=42):
    pooled = np.concatenate(
        [_finite_values(analysis[row_key], metric_key) for analysis in analyses.values()]
    )
    return np.linspace(0.0, _nice_upper(pooled), bins + 1)


def _plot_fraction_histogram(ax, values, edges, color, label):
    clipped = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
    weights = np.full(clipped.size, 1.0 / max(1, clipped.size), dtype=float)
    ax.hist(
        clipped,
        bins=edges,
        weights=weights,
        histtype="step",
        linewidth=2.4,
        color=color,
        label=f"{label} (median {np.median(values):.1f} mm)",
    )


def plot_centroid_separation_distributions(analyses, output_stem):
    metrics = (
        ("min_abs_delta_mu_x_mm", r"minimum pairwise $|\Delta\mu_x|$ [mm]"),
        ("min_abs_delta_mu_y_mm", r"minimum pairwise $|\Delta\mu_y|$ [mm]"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), layout="constrained")
    for axis_index, (metric, xlabel) in enumerate(metrics):
        ax = axes[axis_index]
        edges = _distribution_edges(analyses, "event_rows", metric)
        for sample_index, (label, analysis) in enumerate(analyses.items()):
            values = _finite_values(analysis["event_rows"], metric)
            _plot_fraction_histogram(
                ax,
                values,
                edges,
                COLORS[sample_index % len(COLORS)],
                label,
            )
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("fraction of events per bin", fontsize=15)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelsize=13)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="sample", fontsize=12, title_fontsize=12)
        ax.set_title("x separation" if axis_index == 0 else "y separation", fontsize=17)
    fig.suptitle(
        "Closest truth-shower centroid separation per event",
        fontsize=20,
    )
    fig.text(
        0.5,
        -0.015,
        "Each final histogram bin includes the upper 0.5% tail; centroids use reconstructed-energy × truth-fraction weights.",
        ha="center",
        fontsize=10,
    )
    return _save_figure(fig, output_stem)


def plot_shower_width_distributions(analyses, output_stem):
    metrics = (
        ("sigma_x_mm", r"$\sigma_x$ [mm]", "x width"),
        ("sigma_y_mm", r"$\sigma_y$ [mm]", "y width"),
        ("sigma_r_mm", r"$\sigma_r=\sqrt{\sigma_x^2+\sigma_y^2}$ [mm]", "radial width"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.6), layout="constrained")
    for axis_index, (metric, xlabel, title) in enumerate(metrics):
        ax = axes[axis_index]
        edges = _distribution_edges(analyses, "shower_rows", metric)
        for sample_index, (label, analysis) in enumerate(analyses.items()):
            values = _finite_values(analysis["shower_rows"], metric)
            _plot_fraction_histogram(
                ax,
                values,
                edges,
                COLORS[sample_index % len(COLORS)],
                label,
            )
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("fraction of truth showers per bin", fontsize=14)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(labelsize=12)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="sample", fontsize=11, title_fontsize=11)
        ax.set_title(title, fontsize=16)
    fig.suptitle("Energy-fraction-weighted transverse shower widths", fontsize=20)
    fig.text(
        0.5,
        -0.015,
        "Each final histogram bin includes the upper 0.5% tail; one observation is one truth origin in one event.",
        ha="center",
        fontsize=10,
    )
    return _save_figure(fig, output_stem)


def _secondary_z_axis(ax, layer_z):
    layer_numbers = np.arange(1, layer_z.size + 1, dtype=float)

    def layer_to_z(values):
        return np.interp(np.asarray(values, dtype=float), layer_numbers, layer_z)

    def z_to_layer(values):
        return np.interp(np.asarray(values, dtype=float), layer_z, layer_numbers)

    secondary = ax.secondary_xaxis("top", functions=(layer_to_z, z_to_layer))
    tick_layers = np.unique(
        np.clip(np.asarray([1, 5, 9, 13, 17, 21, 25, 29, layer_z.size]), 1, layer_z.size)
    )
    secondary.set_xticks(layer_z[tick_layers - 1])
    secondary.set_xticklabels([f"{value:.0f}" for value in layer_z[tick_layers - 1]])
    secondary.set_xlabel("physical ECal z [mm]", fontsize=13, labelpad=8)
    secondary.tick_params(labelsize=11)


def plot_longitudinal_profiles(analyses, output_stem):
    reference_z = None
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), layout="constrained")
    metric_specs = (
        (
            "mean_hits_per_event",
            "hit_mean_ci95_low",
            "hit_mean_ci95_high",
            "mean ECal hits per event",
            "Longitudinal hit development",
        ),
        (
            "mean_reconstructed_energy_per_event_mev",
            "energy_mean_ci95_low",
            "energy_mean_ci95_high",
            "mean reconstructed energy per event [MeV]",
            "Longitudinal energy development",
        ),
    )
    for sample_index, (label, analysis) in enumerate(analyses.items()):
        rows = analysis["layer_rows"]
        layer = np.asarray([int(row["layer"]) for row in rows], dtype=float)
        layer_z = np.asarray([float(row["z_mm"]) for row in rows], dtype=float)
        if reference_z is None:
            reference_z = layer_z
        elif reference_z.shape != layer_z.shape or not np.allclose(reference_z, layer_z):
            raise ValueError("Samples do not share the same physical ECal z layers.")
        color = COLORS[sample_index % len(COLORS)]
        for ax, (mean_key, low_key, high_key, ylabel, title) in zip(axes, metric_specs):
            mean = np.asarray([float(row[mean_key]) for row in rows])
            low = np.asarray([float(row[low_key]) for row in rows])
            high = np.asarray([float(row[high_key]) for row in rows])
            ax.plot(
                layer,
                mean,
                marker="o",
                markersize=4,
                linewidth=2.2,
                color=color,
                label=label,
            )
            ax.fill_between(layer, low, high, color=color, alpha=0.17)
            ax.set_xlabel("ECal layer", fontsize=15)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(title, fontsize=17)
            ax.set_xlim(1, layer[-1])
            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.25)
            ax.legend(title="sample", fontsize=12, title_fontsize=12)
    for ax in axes:
        _secondary_z_axis(ax, reference_z)
    event_count = min(analysis["num_events"] for analysis in analyses.values())
    fig.suptitle(
        f"Average event development through the ECal ({event_count:,} events per sample)",
        fontsize=20,
    )
    return _save_figure(fig, output_stem)


def _descriptive_statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "minimum": float(values.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(values.max()),
    }


def _build_summary(analyses):
    event_metrics = (
        "min_abs_delta_mu_x_mm",
        "min_abs_delta_mu_y_mm",
        "mean_abs_delta_mu_x_mm",
        "mean_abs_delta_mu_y_mm",
    )
    shower_metrics = ("sigma_x_mm", "sigma_y_mm", "sigma_r_mm")
    return {
        "scope": "model-independent deterministic 20k cache-prefix dataset characterization",
        "centroid_and_width_weight": "reconstructed ECal hit energy multiplied by simulated physical-origin energy fraction",
        "radial_width_definition": "sigma_r = sqrt(sigma_x^2 + sigma_y^2)",
        "three_electron_reduction": {
            "minimum": "minimum over the three pairwise axis separations in each event",
            "mean": "arithmetic mean over the three pairwise axis separations in each event",
        },
        "samples": {
            label: {
                "cache_dir": analysis["cache_dir"],
                "num_events": analysis["num_events"],
                "num_truth_showers": len(analysis["shower_rows"]),
                "num_origin_count_mismatches": analysis["num_origin_count_mismatches"],
                "event_geometry": {
                    metric: _descriptive_statistics(
                        _finite_values(analysis["event_rows"], metric)
                    )
                    for metric in event_metrics
                },
                "shower_geometry": {
                    metric: _descriptive_statistics(
                        _finite_values(analysis["shower_rows"], metric)
                    )
                    for metric in shower_metrics
                },
            }
            for label, analysis in analyses.items()
        },
    }


def _write_readme(output_dir, analyses, summary):
    lines = [
        "# Physical ECal geometry distributions for the 20k samples",
        "",
        "This bundle is a model-independent description of the physical problem setup.",
        "It scans the same deterministic first 20,000 tensor-cache events per sample used",
        "to construct the supervisor-demo datasets, but it does not load a checkpoint, run",
        "inference, or use train/validation/test membership.",
        "",
        "## Plot index",
        "",
        "1. `01_truth_centroid_separation_distributions` shows the minimum pairwise",
        "   truth-shower centroid separation in x and y for each event. For 2e this is",
        "   the only pair. For 3e it is the closest of the three pairs.",
        "2. `02_truth_shower_width_distributions` shows per-origin transverse widths",
        "   sigma_x, sigma_y, and sigma_r. One observation is one truth shower in one event.",
        "3. `03_longitudinal_hit_and_energy_profiles` shows mean hit and reconstructed-energy",
        "   development through the ECal. The lower axis is layer number and the upper axis",
        "   gives the corresponding physical detector z coordinate in millimetres.",
        "",
        "Both histogram figures use shared binning for 2e and 3e. The displayed upper",
        "limit is the pooled 99.5th percentile; the final bin includes the remaining upper",
        "tail rather than discarding it. Full unbinned values are retained in `data/`.",
        "",
        "## Definitions",
        "",
        "For hit h and physical truth origin i, the weight is",
        "`w_hi = reconstructed_hit_energy_h * simulated_origin_fraction_hi`. The origin",
        "centroid and widths are calculated from these weights:",
        "",
        "- `mu_xi = sum_h(w_hi * x_h) / sum_h(w_hi)` and likewise for y.",
        "- `sigma_xi` and `sigma_yi` are weighted coordinate standard deviations.",
        "- `sigma_ri = sqrt(sigma_xi^2 + sigma_yi^2)`.",
        "",
        "Noise-only hits have zero physical-origin fraction and therefore do not enter",
        "the centroid or width moments. They do enter the detector-level hit and energy",
        "profiles, which describe the full reconstructed ECal event.",
        "",
        "These centroids are intentionally physical, energy-fraction-weighted shower",
        "centroids. They are not the unweighted dominant-origin y centroids used to bind",
        "canonical class labels in the current classifier, and they are not TPad centroids.",
        "The TPad supplies a separate one-dimensional y-like centroid coordinate.",
        "",
        "For 3e, `data/*_event_geometry.csv` also stores the mean pairwise delta-mu in",
        "each axis. The machine-readable distribution statistics are in `summary.json`.",
        "",
        "## Compact numerical summary",
        "",
    ]
    for label in analyses:
        sample = summary["samples"][label]
        event = sample["event_geometry"]
        shower = sample["shower_geometry"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Events: {sample['num_events']:,}; truth showers: {sample['num_truth_showers']:,}.",
                "- Minimum pairwise separation median [IQR]: "
                f"x={event['min_abs_delta_mu_x_mm']['median']:.2f} "
                f"[{event['min_abs_delta_mu_x_mm']['p25']:.2f}, {event['min_abs_delta_mu_x_mm']['p75']:.2f}] mm; "
                f"y={event['min_abs_delta_mu_y_mm']['median']:.2f} "
                f"[{event['min_abs_delta_mu_y_mm']['p25']:.2f}, {event['min_abs_delta_mu_y_mm']['p75']:.2f}] mm.",
                "- Radial shower width median [IQR]: "
                f"{shower['sigma_r_mm']['median']:.2f} "
                f"[{shower['sigma_r_mm']['p25']:.2f}, {shower['sigma_r_mm']['p75']:.2f}] mm.",
                "",
            ]
        )
    path = Path(output_dir) / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    args = parse_args()
    if args.max_events <= 0:
        raise ValueError("--max-events must be positive.")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative.")
    sample_specs = (
        [(label, Path(cache_dir)) for label, cache_dir in args.sample]
        if args.sample
        else list(DEFAULT_SAMPLES)
    )
    if len({label for label, _cache_dir in sample_specs}) != len(sample_specs):
        raise ValueError("Sample labels must be unique.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analyses = {}
    generated = []
    for label, cache_dir in sample_specs:
        analysis = _scan_sample(
            label,
            cache_dir,
            max_events=args.max_events,
            progress_every=args.progress_every,
        )
        analyses[label] = analysis
        generated.extend(
            [
                _write_csv(
                    output_dir / f"data/{label}_event_geometry.csv",
                    analysis["event_rows"],
                ),
                _write_csv(
                    output_dir / f"data/{label}_shower_geometry.csv",
                    analysis["shower_rows"],
                ),
                _write_csv(
                    output_dir / f"data/{label}_layer_profiles.csv",
                    analysis["layer_rows"],
                ),
            ]
        )

    generated.extend(
        plot_centroid_separation_distributions(
            analyses,
            output_dir / "01_truth_centroid_separation_distributions",
        )
    )
    generated.extend(
        plot_shower_width_distributions(
            analyses,
            output_dir / "02_truth_shower_width_distributions",
        )
    )
    generated.extend(
        plot_longitudinal_profiles(
            analyses,
            output_dir / "03_longitudinal_hit_and_energy_profiles",
        )
    )
    summary = _build_summary(analyses)
    summary_path = output_dir / "summary.json"
    save_json(summary_path, summary)
    generated.append(summary_path)
    generated.append(_write_readme(output_dir, analyses, summary))

    manifest_path = output_dir / "manifest.json"
    save_json(
        manifest_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "max_events_per_sample": args.max_events,
            "samples": {
                label: {
                    "cache_dir": analysis["cache_dir"],
                    "num_events": analysis["num_events"],
                    "num_truth_showers": len(analysis["shower_rows"]),
                }
                for label, analysis in analyses.items()
            },
            "generated_files": sorted(
                [str(Path(path).resolve().relative_to(output_dir)) for path in generated]
                + [manifest_path.name]
            ),
        },
    )
    print(f"Saved {len(generated) + 1} artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
