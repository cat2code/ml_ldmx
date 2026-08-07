"""Develop ceiling, centroid-ordering, and detector-context thesis figures."""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.datasets.ecal_tpad_shards import ShardedECalTpadDataset
from ml_ldmx.eval.event_diagnostics import origin_centroid_axis_gap_summary
from ml_ldmx.eval.run_comparison import mean_confidence_interval, quantile_edges
from ml_ldmx.io.artifacts import save_json
from ml_ldmx.viz.training import plot_global_label_swap_recovery


DEFAULT_SAMPLES = (
    (
        "2e",
        PROJECT_ROOT
        / "outputs/supervisor_demo_transformer_20k/transformer_2e_20k_seed7",
        PROJECT_ROOT / "data/ldmx_overlay_events_700k_shards_log1p/2e/events",
    ),
    (
        "3e",
        PROJECT_ROOT
        / "outputs/supervisor_demo_transformer_20k/transformer_3e_20k_seed7",
        PROJECT_ROOT / "data/ldmx_overlay_events_700k_shards_log1p/3e/events",
    ),
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/plots_development/005_ceiling_geometry_detector_context"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="append",
        nargs=3,
        metavar=("LABEL", "RUN_DIR", "CACHE_DIR"),
        help="Sample label, trained run, and sharded tensor cache. Repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional development limit applied after split-position ordering.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["records"] if isinstance(payload, dict) else payload


def _tensor_numpy(value):
    return value.detach().cpu().numpy()


def _scan_sample(label, run_dir, cache_dir, max_events=None):
    reference_path = (
        Path(run_dir)
        / "ceiling_analysis/best/val/reference_event_accuracy.json"
    )
    reference_records = _records(reference_path)
    reference_records = sorted(reference_records, key=lambda row: row["split_position"])
    if max_events is not None:
        reference_records = reference_records[: int(max_events)]
    reference_by_event = {
        int(record["event_idx"]): record for record in reference_records
    }

    dataset = ShardedECalTpadDataset(
        cache_dir,
        max_events=20_000,
        shard_cache_size=1,
    )
    ordered_indices = dataset.order_indices_for_access(reference_by_event)

    geometry_records = []
    positions = []
    energies = []
    layer_event_rows = []
    event_scale_rows = []
    for event_idx in ordered_indices:
        event = dataset[event_idx]
        pos = _tensor_numpy(event["ecal_pos"]).astype(np.float64, copy=False)
        energy = _tensor_numpy(event["ecal_raw_energy"]).astype(np.float64, copy=False)
        labels = event.get("origin_id_y", event.get("physical_y", event["y"]))
        axis_summary = origin_centroid_axis_gap_summary(
            event["ecal_pos"][:, :2],
            labels,
        )
        reference = reference_by_event[int(event_idx)]
        record = {
            "sample": label,
            "event_idx": int(event_idx),
            "split_position": int(reference["split_position"]),
            "accuracy": float(reference["accuracy"]),
            "permutation_invariant_accuracy": float(
                reference["permutation_invariant_accuracy"]
            ),
            "label_permutation_gain": float(reference["label_permutation_gain"]),
            **axis_summary,
        }
        geometry_records.append(record)

        positions.append(pos[:, :2].astype(np.float32, copy=False))
        energies.append(energy.astype(np.float32, copy=False))
        unique_z, inverse = np.unique(pos[:, 2], return_inverse=True)
        hit_counts = np.bincount(inverse, minlength=unique_z.size)
        energy_sums = np.bincount(
            inverse,
            weights=np.clip(energy, 0.0, None),
            minlength=unique_z.size,
        )
        layer_event_rows.append(
            {
                float(z_value): (int(hit_count), float(energy_sum))
                for z_value, hit_count, energy_sum in zip(
                    unique_z,
                    hit_counts,
                    energy_sums,
                )
            }
        )

        tpad = event.get("tpad")
        tpad_count = 0 if tpad is None else int(tpad.shape[0])
        event_scale_rows.append(
            {
                "event_idx": int(event_idx),
                "num_hits": int(pos.shape[0]),
                "total_reconstructed_energy": float(np.clip(energy, 0.0, None).sum()),
                "num_tpad_tokens": tpad_count,
                "min_origin_centroid_gap_y": axis_summary[
                    "min_origin_centroid_gap_y"
                ],
            }
        )

    geometry_records.sort(key=lambda row: row["split_position"])
    event_scale_rows.sort(
        key=lambda row: reference_by_event[row["event_idx"]]["split_position"]
    )
    all_z = sorted({z_value for row in layer_event_rows for z_value in row})
    layer_hits = np.zeros((len(layer_event_rows), len(all_z)), dtype=float)
    layer_energy = np.zeros_like(layer_hits)
    z_to_index = {z_value: index for index, z_value in enumerate(all_z)}
    for event_row, values in enumerate(layer_event_rows):
        for z_value, (hit_count, energy_sum) in values.items():
            layer_index = z_to_index[z_value]
            layer_hits[event_row, layer_index] = hit_count
            layer_energy[event_row, layer_index] = energy_sum

    return {
        "label": label,
        "run_dir": str(Path(run_dir).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "reference_records": reference_records,
        "geometry_records": geometry_records,
        "positions_xy": np.concatenate(positions, axis=0),
        "energies": np.concatenate(energies, axis=0),
        "layer_z": np.asarray(all_z, dtype=float),
        "layer_hits": layer_hits,
        "layer_energy": layer_energy,
        "event_scale_rows": event_scale_rows,
    }


def _save_figure(fig, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _masked_positive(values, minimum=0.0):
    return np.ma.masked_less_equal(np.asarray(values, dtype=float), float(minimum))


def _binned_profile(records, x_key, y_key, bootstrap_samples, seed, num_bins=8):
    pairs = np.asarray(
        [
            (float(row[x_key]), float(row[y_key]))
            for row in records
            if row.get(x_key) is not None
            and row.get(y_key) is not None
            and np.isfinite(float(row[x_key]))
            and np.isfinite(float(row[y_key]))
        ],
        dtype=float,
    )
    if pairs.size == 0:
        return []
    edges = quantile_edges(pairs[:, 0], num_bins=num_bins)
    if edges is None:
        return []
    output = []
    for bin_index in range(edges.size - 1):
        selected = (
            (pairs[:, 0] >= edges[bin_index])
            & (
                pairs[:, 0] <= edges[bin_index + 1]
                if bin_index == edges.size - 2
                else pairs[:, 0] < edges[bin_index + 1]
            )
        )
        if not selected.any():
            continue
        mean, low, high, _method = mean_confidence_interval(
            pairs[selected, 1],
            bootstrap_samples=bootstrap_samples,
            seed=seed + bin_index,
        )
        output.append(
            {
                "x": float(pairs[selected, 0].mean()),
                "mean": float(mean),
                "low": float(low),
                "high": float(high),
                "count": int(selected.sum()),
            }
        )
    return output


def plot_centroid_axis_profiles(analyses, output_stem, bootstrap_samples=300, seed=7):
    metric_specs = (
        (
            "x_gap_for_y_closest_origin_pair",
            r"$|\Delta x|$ for the pair closest in $y$ [mm]",
        ),
        (
            "min_origin_centroid_gap_y",
            r"minimum canonical-ordering centroid gap $|\Delta y|$ [mm]",
        ),
    )
    labels = list(analyses)
    x_limits = {}
    count_max = 1.0
    cached = {}
    for metric, _xlabel in metric_specs:
        all_x = np.concatenate(
            [
                np.asarray(
                    [float(row[metric]) for row in analyses[label]["geometry_records"]],
                    dtype=float,
                )
                for label in labels
            ]
        )
        x_limits[metric] = max(1.0, float(np.quantile(all_x, 0.995)))
        for label in labels:
            records = analyses[label]["geometry_records"]
            x = np.asarray([float(row[metric]) for row in records])
            y = np.asarray([float(row["accuracy"]) for row in records])
            x_edges = np.linspace(0.0, x_limits[metric], 31)
            y_edges = np.linspace(0.0, 1.0, 31)
            visible = x <= x_edges[-1]
            counts, _, _ = np.histogram2d(
                np.clip(x[visible], x_edges[0], np.nextafter(x_edges[-1], 0.0)),
                y[visible],
                bins=(x_edges, y_edges),
            )
            cached[(label, metric)] = (x_edges, y_edges, counts)
            count_max = max(count_max, float(counts.max()))

    fig, axes = plt.subplots(
        len(labels),
        len(metric_specs),
        figsize=(15.5, 5.5 * len(labels)),
        squeeze=False,
        layout="constrained",
    )
    meshes = []
    for row_index, label in enumerate(labels):
        records = analyses[label]["geometry_records"]
        for column_index, (metric, xlabel) in enumerate(metric_specs):
            ax = axes[row_index, column_index]
            x_edges, y_edges, counts = cached[(label, metric)]
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_less(counts.T, 1.0),
                cmap="viridis",
                norm=LogNorm(vmin=1.0, vmax=count_max),
                shading="flat",
            )
            meshes.append(mesh)
            for target, target_label, color, offset in (
                ("accuracy", "ordinary accuracy", "#2563eb", 0),
                (
                    "permutation_invariant_accuracy",
                    "after optimal global relabeling",
                    "#d97706",
                    100,
                ),
            ):
                profile = _binned_profile(
                    records,
                    metric,
                    target,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + row_index * 1000 + column_index * 100 + offset,
                )
                x_values = np.asarray([point["x"] for point in profile])
                means = np.asarray([point["mean"] for point in profile])
                lows = np.asarray([point["low"] for point in profile])
                highs = np.asarray([point["high"] for point in profile])
                visible = x_values <= x_edges[-1]
                ax.plot(
                    x_values[visible],
                    means[visible],
                    marker="o",
                    linewidth=2.2,
                    color=color,
                    label=target_label,
                    zorder=4,
                )
                ax.fill_between(
                    x_values[visible],
                    lows[visible],
                    highs[visible],
                    color=color,
                    alpha=0.16,
                    zorder=3,
                )
            ax.set_xlim(0.0, x_edges[-1])
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(xlabel, fontsize=15)
            ax.set_ylabel("event hit accuracy", fontsize=15)
            ax.set_title(f"{label} validation events", fontsize=17)
            ax.tick_params(labelsize=13)
            ax.legend(fontsize=12, loc="lower right")
    fig.colorbar(
        meshes[-1],
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
        label="events per populated rectangular bin",
    )
    fig.suptitle(
        "Does canonical-y centroid closeness explain low event accuracy?",
        fontsize=20,
    )
    return _save_figure(fig, output_stem)


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.size < 2 or np.std(first) <= 0.0 or np.std(second) <= 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _partial_spearman(target, predictor, control):
    target_rank = _rankdata(target)
    predictor_rank = _rankdata(predictor)
    control_rank = _rankdata(control)
    design = np.column_stack([np.ones(control_rank.size), control_rank])
    target_residual = target_rank - design @ np.linalg.lstsq(
        design, target_rank, rcond=None
    )[0]
    predictor_residual = predictor_rank - design @ np.linalg.lstsq(
        design, predictor_rank, rcond=None
    )[0]
    return _correlation(target_residual, predictor_residual)


def centroid_axis_statistics(analysis):
    records = analysis["geometry_records"]
    x_gap = np.asarray(
        [float(row["x_gap_for_y_closest_origin_pair"]) for row in records]
    )
    y_gap = np.asarray([float(row["min_origin_centroid_gap_y"]) for row in records])
    output = {"num_events": len(records)}
    for target in (
        "accuracy",
        "permutation_invariant_accuracy",
        "label_permutation_gain",
    ):
        values = np.asarray([float(row[target]) for row in records])
        target_rank = _rankdata(values)
        x_rank = _rankdata(x_gap)
        y_rank = _rankdata(y_gap)
        output[target] = {
            "spearman_x_gap_same_pair": _correlation(target_rank, x_rank),
            "spearman_y_gap": _correlation(target_rank, y_rank),
            "partial_spearman_x_controlling_y": _partial_spearman(
                values, x_gap, y_gap
            ),
            "partial_spearman_y_controlling_x": _partial_spearman(
                values, y_gap, x_gap
            ),
        }
    return output


def plot_centroid_axis_joint_effects(analyses, output_stem):
    labels = list(analyses)
    fig, axes = plt.subplots(
        len(labels),
        2,
        figsize=(14.5, 5.7 * len(labels)),
        squeeze=False,
        layout="constrained",
    )
    column_meshes = [[], []]
    for row_index, label in enumerate(labels):
        records = analyses[label]["geometry_records"]
        x_gap = np.asarray(
            [float(row["x_gap_for_y_closest_origin_pair"]) for row in records]
        )
        y_gap = np.asarray(
            [float(row["min_origin_centroid_gap_y"]) for row in records]
        )
        x_edges = np.linspace(0.0, max(1.0, float(np.quantile(x_gap, 0.99))), 13)
        y_edges = np.linspace(0.0, max(1.0, float(np.quantile(y_gap, 0.99))), 13)
        x_bin = np.clip(np.digitize(x_gap, x_edges) - 1, 0, x_edges.size - 2)
        y_bin = np.clip(np.digitize(y_gap, y_edges) - 1, 0, y_edges.size - 2)
        for column_index, (target, title, cmap, norm) in enumerate(
            (
                (
                    "accuracy",
                    "mean ordinary event accuracy",
                    "viridis",
                    Normalize(vmin=0.45, vmax=0.95),
                ),
                (
                    "label_permutation_gain",
                    "mean accuracy recovered by global relabeling",
                    "magma",
                    Normalize(vmin=0.0, vmax=0.12),
                ),
            )
        ):
            sums = np.zeros((x_edges.size - 1, y_edges.size - 1), dtype=float)
            counts = np.zeros_like(sums)
            for event_index, record in enumerate(records):
                sums[x_bin[event_index], y_bin[event_index]] += float(record[target])
                counts[x_bin[event_index], y_bin[event_index]] += 1.0
            means = np.divide(
                sums,
                counts,
                out=np.full_like(sums, np.nan),
                where=counts >= 5,
            )
            ax = axes[row_index, column_index]
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(means.T),
                cmap=cmap,
                norm=norm,
                shading="flat",
            )
            column_meshes[column_index].append(mesh)
            ax.set_xlabel(
                r"$|\Delta x|$ for the pair closest in $y$ [mm]",
                fontsize=14,
            )
            ax.set_ylabel(
                r"minimum canonical-ordering gap $|\Delta y|$ [mm]",
                fontsize=14,
            )
            ax.set_title(f"{label}: {title}", fontsize=16)
            ax.tick_params(labelsize=12)
    for column_index, label in enumerate(
        ("mean event accuracy", "mean recovered accuracy")
    ):
        fig.colorbar(
            column_meshes[column_index][-1],
            ax=axes[:, column_index].ravel().tolist(),
            fraction=0.04,
            pad=0.02,
            label=label,
        )
    fig.suptitle(
        "Joint x/y centroid-gap dependence for the y-ordering-critical pair",
        fontsize=20,
    )
    return _save_figure(fig, output_stem)


def plot_detector_xy_context(analyses, output_stem):
    labels = list(analyses)
    all_positions = np.concatenate(
        [analyses[label]["positions_xy"] for label in labels],
        axis=0,
    )
    robust_limit = float(np.quantile(np.abs(all_positions), 0.998))
    limit = max(25.0, math.ceil(robust_limit / 25.0) * 25.0)
    edges = np.linspace(-limit, limit, 65)
    histograms = {}
    count_max = 1.0
    energy_max = 1.0
    energy_min = math.inf
    for label in labels:
        positions = analyses[label]["positions_xy"]
        energy = analyses[label]["energies"]
        counts, _, _ = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=(edges, edges),
        )
        energy_sum, _, _ = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=(edges, edges),
            weights=np.clip(energy, 0.0, None),
        )
        histograms[label] = (counts, energy_sum)
        count_max = max(count_max, float(counts.max()))
        energy_max = max(energy_max, float(energy_sum.max()))
        positive_energy = energy_sum[energy_sum > 0.0]
        if positive_energy.size:
            energy_min = min(energy_min, float(positive_energy.min()))

    fig, axes = plt.subplots(
        len(labels),
        2,
        figsize=(13.5, 5.8 * len(labels)),
        squeeze=False,
        layout="constrained",
    )
    meshes = [[], []]
    for row_index, label in enumerate(labels):
        counts, energy_sum = histograms[label]
        for column_index, (values, title, norm, colorbar_label) in enumerate(
            (
                (
                    counts,
                    "ECal hit occupancy",
                    LogNorm(vmin=1.0, vmax=count_max),
                    "hit occurrences per populated bin",
                ),
                (
                    energy_sum,
                    "reconstructed ECal energy density",
                    LogNorm(vmin=max(1e-6, energy_min), vmax=energy_max),
                    "summed reconstructed energy per populated bin [MeV]",
                ),
            )
        ):
            ax = axes[row_index, column_index]
            mesh = ax.pcolormesh(
                edges,
                edges,
                _masked_positive(values.T),
                cmap="viridis",
                norm=norm,
                shading="flat",
            )
            meshes[column_index].append(mesh)
            ax.set_aspect("equal")
            ax.set_xlabel("ECal x [mm]", fontsize=14)
            ax.set_ylabel("ECal y [mm]", fontsize=14)
            ax.set_title(f"{label}: {title}", fontsize=16)
            ax.tick_params(labelsize=12)
            if row_index == len(labels) - 1:
                fig.colorbar(
                    meshes[column_index][-1],
                    ax=axes[:, column_index].ravel().tolist(),
                    fraction=0.04,
                    pad=0.02,
                    label=colorbar_label,
                )
    fig.suptitle(
        "Where validation events deposit hits and energy in the ECal",
        fontsize=20,
    )
    return _save_figure(fig, output_stem)


def _mean_and_ci(matrix):
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    if matrix.shape[0] <= 1:
        return mean, mean, mean
    sem = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def plot_detector_longitudinal_context(analyses, output_stem):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), layout="constrained")
    colors = ("#2563eb", "#d97706", "#0f766e", "#7c3aed")
    for sample_index, (label, analysis) in enumerate(analyses.items()):
        layer_numbers = np.arange(1, analysis["layer_z"].size + 1)
        for ax, matrix, ylabel in (
            (axes[0], analysis["layer_hits"], "mean ECal hits per event"),
            (
                axes[1],
                analysis["layer_energy"],
                "mean reconstructed energy per event [MeV]",
            ),
        ):
            mean, low, high = _mean_and_ci(matrix)
            color = colors[sample_index % len(colors)]
            ax.plot(
                layer_numbers,
                mean,
                marker="o",
                markersize=4,
                linewidth=2.2,
                color=color,
                label=label,
            )
            ax.fill_between(layer_numbers, low, high, color=color, alpha=0.17)
            ax.set_xlabel("ECal layer", fontsize=15)
            ax.set_ylabel(ylabel, fontsize=15)
            ax.tick_params(labelsize=13)
            ax.grid(True, alpha=0.25)
            ax.legend(title="sample", fontsize=12)
    axes[0].set_title("Longitudinal hit development", fontsize=17)
    axes[1].set_title("Longitudinal energy development", fontsize=17)
    fig.suptitle("Average validation-event development through the ECal", fontsize=20)
    return _save_figure(fig, output_stem)


def plot_event_scale_context(analyses, output_stem):
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), layout="constrained")
    colors = ("#2563eb", "#d97706", "#0f766e", "#7c3aed")
    for sample_index, (label, analysis) in enumerate(analyses.items()):
        rows = analysis["event_scale_rows"]
        hits = np.asarray([row["num_hits"] for row in rows], dtype=float)
        energy = np.asarray(
            [row["total_reconstructed_energy"] for row in rows], dtype=float
        )
        color = colors[sample_index % len(colors)]
        axes[0].hist(
            hits,
            bins=40,
            histtype="step",
            linewidth=2.2,
            color=color,
            label=label,
        )
        axes[1].hist(
            energy,
            bins=40,
            histtype="step",
            linewidth=2.2,
            color=color,
            label=label,
        )
        token_counts = np.asarray([row["num_tpad_tokens"] for row in rows], dtype=int)
        unique, counts = np.unique(token_counts, return_counts=True)
        width = 0.34
        offset = (sample_index - 0.5 * (len(analyses) - 1)) * width
        axes[2].bar(
            unique + offset,
            counts,
            width=width,
            color=color,
            alpha=0.82,
            label=label,
        )
    for ax in axes:
        ax.set_ylabel("events", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="sample", fontsize=12)
    axes[0].set_xlabel("ECal hits per event", fontsize=14)
    axes[1].set_xlabel("total reconstructed ECal energy [MeV]", fontsize=14)
    axes[2].set_xlabel("available TPad tokens per event", fontsize=14)
    axes[0].set_title("Hit multiplicity", fontsize=16)
    axes[1].set_title("Event energy", fontsize=16)
    axes[2].set_title("TPad context multiplicity", fontsize=16)
    fig.suptitle("Validation-event scale before model evaluation", fontsize=20)
    return _save_figure(fig, output_stem)


def _representative_event_idx(analysis):
    rows = analysis["event_scale_rows"]
    metrics = np.asarray(
        [
            [
                float(row["num_hits"]),
                float(row["total_reconstructed_energy"]),
                float(row["min_origin_centroid_gap_y"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    median = np.median(metrics, axis=0)
    scale = np.maximum(np.subtract(*np.quantile(metrics, [0.75, 0.25], axis=0)), 1e-9)
    score = np.sum(np.abs(metrics - median) / scale, axis=1)
    return int(rows[int(np.argmin(score))]["event_idx"])


def plot_truth_event_examples(analyses, output_stem):
    fig, axes = plt.subplots(
        len(analyses),
        2,
        figsize=(14.0, 5.6 * len(analyses)),
        squeeze=False,
        layout="constrained",
    )
    class_colors = ("#2563eb", "#d97706", "#0f766e", "#7c3aed")
    for row_index, (label, analysis) in enumerate(analyses.items()):
        event_idx = _representative_event_idx(analysis)
        dataset = ShardedECalTpadDataset(
            analysis["cache_dir"],
            max_events=20_000,
            shard_cache_size=1,
        )
        event = dataset[event_idx]
        pos = _tensor_numpy(event["ecal_pos"]).astype(float)
        energy = np.clip(_tensor_numpy(event["ecal_raw_energy"]).astype(float), 0.0, None)
        labels = _tensor_numpy(
            event.get("origin_id_y", event.get("physical_y", event["y"]))
        ).astype(int)
        marker_size = 12.0 + 55.0 * np.log1p(energy) / max(1e-9, np.log1p(energy).max())
        for class_index, truth_label in enumerate(sorted(set(labels.tolist()))):
            selected = labels == truth_label
            color = class_colors[class_index % len(class_colors)]
            axes[row_index, 0].scatter(
                pos[selected, 0],
                pos[selected, 1],
                s=marker_size[selected],
                alpha=0.62,
                color=color,
                edgecolors="none",
                label=f"truth origin {truth_label}",
            )
            axes[row_index, 1].scatter(
                pos[selected, 2],
                pos[selected, 1],
                s=marker_size[selected],
                alpha=0.62,
                color=color,
                edgecolors="none",
                label=f"truth origin {truth_label}",
            )
        tpad = event.get("tpad")
        if tpad is not None and int(tpad.shape[0]) > 0:
            tpad_y = _tensor_numpy(tpad)[:, 0]
            z_marker = float(pos[:, 2].min() - 0.06 * np.ptp(pos[:, 2]))
            axes[row_index, 1].scatter(
                np.full(tpad_y.shape, z_marker),
                tpad_y,
                marker=">",
                s=90,
                color="#111827",
                label="TPad centroid",
                zorder=5,
            )
        axes[row_index, 0].set_xlabel("ECal x [mm]", fontsize=14)
        axes[row_index, 0].set_ylabel("ECal y [mm]", fontsize=14)
        axes[row_index, 0].set_aspect("equal", adjustable="datalim")
        axes[row_index, 0].set_title(
            f"{label} event {event_idx}: transverse projection",
            fontsize=16,
        )
        axes[row_index, 1].set_xlabel("ECal z [mm]", fontsize=14)
        axes[row_index, 1].set_ylabel("ECal y [mm]", fontsize=14)
        axes[row_index, 1].set_title(
            f"{label} event {event_idx}: longitudinal projection",
            fontsize=16,
        )
        for ax in axes[row_index]:
            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.18)
            ax.legend(fontsize=11, markerscale=0.9)
    fig.suptitle(
        "Representative truth-level event anatomy (marker area follows hit energy)",
        fontsize=20,
    )
    return _save_figure(fig, output_stem)


def _write_readme(output_dir, analyses, statistics):
    lines = [
        "# Ceiling, centroid-ordering, and detector-context development plots",
        "",
        "This bundle uses the saved best-checkpoint validation records for the 20k",
        "supervisor-demo ECalTpadTransformer models. Raw cached validation events are",
        "read only to calculate truth geometry and detector-context summaries; the",
        "models are not rerun or retrained.",
        "",
        "## Plot index",
        "",
        "1. `01_*_global_label_swap_recovery` extracts the upper-left ceiling",
        "   diagnostic as a thesis-sized standalone plot. Its vertical axis begins",
        "   at the first populated accuracy region rather than zero.",
        "2. `02_centroid_axis_accuracy_profiles` compares the minimum canonical-y",
        "   centroid gap with the x gap of the same origin pair. It shows ordinary",
        "   and globally relabeled accuracy together.",
        "3. `03_centroid_axis_joint_effects` conditions simultaneously on x and y",
        "   gaps. Sparse cells with fewer than five events are left white.",
        "4. `04_detector_xy_context` shows aggregate transverse hit occupancy and",
        "   reconstructed-energy density before any model result.",
        "5. `05_detector_longitudinal_context` shows average hit and energy",
        "   development through the ECal layers.",
        "6. `06_event_scale_context` compares hit counts, total ECal energy, and",
        "   available TPad-token multiplicity.",
        "7. `07_truth_event_examples` gives truth-only transverse and longitudinal",
        "   examples. Marker area follows reconstructed hit energy; black triangles",
        "   show the one-dimensional TPad centroids upstream of the ECal.",
        "8. `rectangular_heatmap_reversion/` regenerates the existing rectangular",
        "   density figures with empty bins white and the count scale beginning at",
        "   one occurrence.",
        "",
        "## TPad context and TPad ablation answer different questions",
        "",
        "`Performance versus available TPad context` is an observational grouping of",
        "the normal, non-ablated evaluation. Complete TPad means that the event has no",
        "token deficit relative to its electron count; missing TPad means fewer tokens",
        "than electrons. It compares ordinary and permutation-invariant hit-weighted",
        "accuracy between those naturally occurring groups. It does not remove tokens,",
        "and group differences can reflect event difficulty rather than a TPad effect.",
        "",
        "The paired TPad-ablation plot evaluates each event twice with the same trained",
        "checkpoint, first normally and then after removing every TPad token. It therefore",
        "tests the checkpoint's immediate reliance on TPad input. It does not test how well",
        "a model trained from scratch without TPad could perform.",
        "",
        "## Canonical-y hypothesis",
        "",
        "The canonical classes are ordered by the unweighted mean ECal y position",
        "of each hard truth origin. The y-gap diagnostic therefore uses exactly those",
        "centroids. For 3e events, the x comparison uses the same origin pair that is",
        "closest in y; this avoids comparing two different pairs.",
        "",
        "Partial Spearman correlations control one axis gap for the other. The full",
        "machine-readable values are in `centroid_axis_statistics.json`.",
        "",
    ]
    for label in analyses:
        ordinary = statistics[label]["accuracy"]
        invariant = statistics[label]["permutation_invariant_accuracy"]
        gain = statistics[label]["label_permutation_gain"]
        lines.extend(
            [
                f"### {label}",
                "",
                "- Ordinary accuracy: partial Spearman "
                f"y|x={ordinary['partial_spearman_y_controlling_x']:+.3f}, "
                f"x|y={ordinary['partial_spearman_x_controlling_y']:+.3f}.",
                "- Permutation-invariant accuracy: partial Spearman "
                f"y|x={invariant['partial_spearman_y_controlling_x']:+.3f}, "
                f"x|y={invariant['partial_spearman_x_controlling_y']:+.3f}.",
                "- Accuracy recovered by relabeling: partial Spearman "
                f"y|x={gain['partial_spearman_y_controlling_x']:+.3f}, "
                f"x|y={gain['partial_spearman_x_controlling_y']:+.3f}.",
                "",
            ]
        )
    lines.extend(
        [
            "The strong positive y-gap association remains after optimal global",
            "relabeling. Therefore close y centroids are associated with genuinely hard",
            "overlapping showers, not only with an arbitrary whole-event class swap.",
            "The negative association between y gap and recovered accuracy shows that",
            "global label binding contributes additionally when the y ordering is close.",
            "For 2e, the x gap adds little after controlling y; for 3e, both axes retain",
            "an association, although y is stronger in this preliminary comparison.",
            "These are correlations rather than a causal decomposition and do not yet",
            "control shower energy, width, or hit multiplicity.",
            "",
        ]
    )
    lines.extend(
        [
            "## Recommended thesis use",
            "",
            "Use the aggregate detector XY figure and longitudinal profile before",
            "the ML results: together they explain the transverse overlap problem and",
            "where showers develop in depth. A truth-only representative event can",
            "then provide a concrete reference. The hit/energy/TPad scale distributions",
            "are useful dataset-characterization material, but are less essential in",
            "the main text and may fit better in an appendix.",
            "",
            "These are development figures based on validation events. Final design",
            "choices should be frozen before producing held-out test-set results.",
            "",
        ]
    )
    path = Path(output_dir) / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    args = parse_args()
    if args.max_events is not None and args.max_events <= 0:
        raise ValueError("--max-events must be positive when provided.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_specs = (
        [(label, Path(run_dir), Path(cache_dir)) for label, run_dir, cache_dir in args.sample]
        if args.sample
        else list(DEFAULT_SAMPLES)
    )

    analyses = {}
    generated = []
    for label, run_dir, cache_dir in sample_specs:
        analysis = _scan_sample(
            label,
            run_dir,
            cache_dir,
            max_events=args.max_events,
        )
        analyses[label] = analysis
        records_path = output_dir / f"data/{label}_centroid_axis_records.json"
        save_json(records_path, analysis["geometry_records"])
        generated.append(records_path)
        for suffix in ("png", "pdf"):
            output_path = output_dir / f"01_{label}_global_label_swap_recovery.{suffix}"
            plot_global_label_swap_recovery(
                analysis["reference_records"],
                output_path,
                subtitle=(
                    f"ECalTpadTransformer, {label} validation "
                    f"(N={len(analysis['reference_records']):,})"
                ),
            )
            generated.append(output_path)

    generated.extend(
        plot_centroid_axis_profiles(
            analyses,
            output_dir / "02_centroid_axis_accuracy_profiles",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    )
    generated.extend(
        plot_centroid_axis_joint_effects(
            analyses,
            output_dir / "03_centroid_axis_joint_effects",
        )
    )
    generated.extend(
        plot_detector_xy_context(
            analyses,
            output_dir / "04_detector_xy_context",
        )
    )
    generated.extend(
        plot_detector_longitudinal_context(
            analyses,
            output_dir / "05_detector_longitudinal_context",
        )
    )
    generated.extend(
        plot_event_scale_context(
            analyses,
            output_dir / "06_event_scale_context",
        )
    )
    generated.extend(
        plot_truth_event_examples(
            analyses,
            output_dir / "07_truth_event_examples",
        )
    )

    statistics = {
        label: centroid_axis_statistics(analysis)
        for label, analysis in analyses.items()
    }
    statistics_path = output_dir / "centroid_axis_statistics.json"
    save_json(statistics_path, statistics)
    generated.append(statistics_path)
    generated.append(_write_readme(output_dir, analyses, statistics))
    manifest_path = output_dir / "manifest.json"
    save_json(
        manifest_path,
        {
            "samples": {
                label: {
                    "run_dir": analysis["run_dir"],
                    "cache_dir": analysis["cache_dir"],
                    "num_events": len(analysis["geometry_records"]),
                }
                for label, analysis in analyses.items()
            },
            "generated_files": [
                str(Path(path).resolve().relative_to(output_dir)) for path in generated
            ],
        },
    )
    print(f"Saved {len(generated) + 1} development artifacts to {output_dir}")


if __name__ == "__main__":
    main()
