"""Generate 20k detector-context density plots and TPad count diagnostics."""

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
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/plots_development/007_detector_context_density_20k"
)
MAX_LAYERS = 64
Y_BIN_WIDTH_MM = 5.0
BASE_Y_EDGES = np.arange(-500.0, 500.0 + Y_BIN_WIDTH_MM, Y_BIN_WIDTH_MM)


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
    }
)


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
    parser.add_argument("--max-events", type=int, default=20_000)
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


def _scan_sample(label, cache_dir, max_events, progress_every):
    dataset = ShardedECalTpadDataset(
        cache_dir,
        max_events=max_events,
        shard_cache_size=1,
    )
    num_events = len(dataset)
    layer_hits = np.zeros((num_events, MAX_LAYERS), dtype=np.uint16)
    layer_energy = np.zeros((num_events, MAX_LAYERS), dtype=np.float32)
    yz_hit_counts = np.zeros((MAX_LAYERS, BASE_Y_EDGES.size - 1), dtype=np.int64)
    yz_energy = np.zeros_like(yz_hit_counts, dtype=np.float64)
    layer_hit_y_sum = np.zeros(MAX_LAYERS, dtype=np.float64)
    layer_hit_count = np.zeros(MAX_LAYERS, dtype=np.int64)
    layer_energy_y_sum = np.zeros(MAX_LAYERS, dtype=np.float64)
    layer_energy_sum = np.zeros(MAX_LAYERS, dtype=np.float64)
    tpad_y_counts = np.zeros(BASE_Y_EDGES.size - 1, dtype=np.int64)
    tpad_counts = np.zeros(num_events, dtype=np.int16)
    electron_counts = np.zeros(num_events, dtype=np.int16)
    z_to_column = {}
    y_underflow_hits = 0
    y_overflow_hits = 0
    tpad_y_underflow = 0
    tpad_y_overflow = 0

    for event_idx in range(num_events):
        event = dataset[event_idx]
        position = _numpy(event["ecal_pos"])
        energy = np.clip(
            _numpy(event["ecal_raw_energy"]).reshape(-1),
            0.0,
            None,
        )
        z_rounded = np.round(position[:, 2], 3)
        unique_z, inverse = np.unique(z_rounded, return_inverse=True)
        columns = np.empty(unique_z.size, dtype=np.int64)
        for local_index, z_value in enumerate(unique_z):
            z_key = float(z_value)
            if z_key not in z_to_column:
                if len(z_to_column) >= MAX_LAYERS:
                    raise ValueError(f"Found more than {MAX_LAYERS} ECal z layers.")
                z_to_column[z_key] = len(z_to_column)
            columns[local_index] = z_to_column[z_key]
        hit_columns = columns[inverse]

        hit_counts = np.bincount(inverse, minlength=unique_z.size)
        energy_sums = np.bincount(
            inverse,
            weights=energy,
            minlength=unique_z.size,
        )
        layer_hits[event_idx, columns] = hit_counts.astype(np.uint16)
        layer_energy[event_idx, columns] = energy_sums.astype(np.float32)

        y = position[:, 1]
        y_bin = np.searchsorted(BASE_Y_EDGES, y, side="right") - 1
        y_underflow_hits += int(np.count_nonzero(y_bin < 0))
        y_overflow_hits += int(np.count_nonzero(y_bin >= BASE_Y_EDGES.size - 1))
        visible = (y_bin >= 0) & (y_bin < BASE_Y_EDGES.size - 1)
        np.add.at(
            yz_hit_counts,
            (hit_columns[visible], y_bin[visible]),
            1,
        )
        np.add.at(
            yz_energy,
            (hit_columns[visible], y_bin[visible]),
            energy[visible],
        )
        np.add.at(layer_hit_y_sum, hit_columns, y)
        np.add.at(layer_hit_count, hit_columns, 1)
        np.add.at(layer_energy_y_sum, hit_columns, energy * y)
        np.add.at(layer_energy_sum, hit_columns, energy)

        tpad = event.get("tpad")
        if tpad is None:
            tpad_y = np.empty(0, dtype=float)
        else:
            tpad_y = _numpy(tpad)[:, 0]
        tpad_counts[event_idx] = int(tpad_y.size)
        electron_counts[event_idx] = _scalar_int(
            event.get("electron_count", int(label.rstrip("e")))
        )
        tpad_bin = np.searchsorted(BASE_Y_EDGES, tpad_y, side="right") - 1
        tpad_y_underflow += int(np.count_nonzero(tpad_bin < 0))
        tpad_y_overflow += int(np.count_nonzero(tpad_bin >= BASE_Y_EDGES.size - 1))
        tpad_visible = (tpad_bin >= 0) & (tpad_bin < BASE_Y_EDGES.size - 1)
        tpad_y_counts += np.bincount(
            tpad_bin[tpad_visible],
            minlength=BASE_Y_EDGES.size - 1,
        )

        if progress_every > 0 and (event_idx + 1) % progress_every == 0:
            print(f"{label}: scanned {event_idx + 1:,}/{num_events:,} events", flush=True)

    ordered_z = np.asarray(sorted(z_to_column), dtype=float)
    order = np.asarray([z_to_column[float(z_value)] for z_value in ordered_z], dtype=int)
    num_layers = ordered_z.size
    layer_hits = layer_hits[:, order]
    layer_energy = layer_energy[:, order]
    yz_hit_counts = yz_hit_counts[order]
    yz_energy = yz_energy[order]
    layer_hit_y_sum = layer_hit_y_sum[order]
    layer_hit_count = layer_hit_count[order]
    layer_energy_y_sum = layer_energy_y_sum[order]
    layer_energy_sum = layer_energy_sum[order]
    return {
        "label": label,
        "cache_dir": str(Path(cache_dir).resolve()),
        "num_events": num_events,
        "layer_z": ordered_z,
        "layer_hits": layer_hits,
        "layer_energy": layer_energy,
        "yz_hit_counts": yz_hit_counts,
        "yz_energy": yz_energy,
        "layer_mean_hit_y": np.divide(
            layer_hit_y_sum,
            layer_hit_count,
            out=np.full(num_layers, np.nan),
            where=layer_hit_count > 0,
        ),
        "layer_mean_energy_y": np.divide(
            layer_energy_y_sum,
            layer_energy_sum,
            out=np.full(num_layers, np.nan),
            where=layer_energy_sum > 0,
        ),
        "layer_hit_count": layer_hit_count,
        "layer_energy_sum": layer_energy_sum,
        "tpad_y_counts": tpad_y_counts,
        "tpad_counts": tpad_counts,
        "electron_counts": electron_counts,
        "range_checks": {
            "ecal_y_underflow_hits": y_underflow_hits,
            "ecal_y_overflow_hits": y_overflow_hits,
            "tpad_y_underflow_tokens": tpad_y_underflow,
            "tpad_y_overflow_tokens": tpad_y_overflow,
        },
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


def _populated(values, minimum=1.0):
    return np.ma.masked_less(np.asarray(values, dtype=float), float(minimum))


def _mean_ci(matrix):
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    if matrix.shape[0] <= 1:
        return mean, mean, mean
    sem = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _draw_mean_curve(ax, x, mean, low, high):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    yerr = np.vstack(
        [np.clip(mean - low, 0.0, None), np.clip(high - mean, 0.0, None)]
    )
    ax.errorbar(
        x,
        mean,
        yerr=yerr,
        fmt="-o",
        color="#111827",
        ecolor="#111827",
        linewidth=4.0,
        elinewidth=3.5,
        capsize=4,
        markersize=6.2,
        zorder=4,
    )
    return ax.errorbar(
        x,
        mean,
        yerr=yerr,
        fmt="-o",
        color="white",
        ecolor="white",
        markerfacecolor="white",
        markeredgecolor="#111827",
        markeredgewidth=0.8,
        linewidth=2.0,
        elinewidth=1.7,
        capsize=2.8,
        markersize=4.8,
        label="event mean ± 95% CI",
        zorder=5,
    )


def _nice_upper(values, quantile=0.995):
    upper = float(np.quantile(np.asarray(values, dtype=float), quantile))
    if upper <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(upper))
    step = magnitude / 2.0
    return max(step, math.ceil(upper / step) * step)


def _layer_density(analyses, matrix_key, y_bins=55):
    pooled = np.concatenate(
        [np.asarray(analysis[matrix_key], dtype=float).reshape(-1) for analysis in analyses.values()]
    )
    upper = _nice_upper(pooled)
    if matrix_key == "layer_hits":
        y_edges = np.arange(-0.5, math.ceil(upper) + 1.5, 1.0)
    else:
        y_edges = np.linspace(0.0, upper, int(y_bins) + 1)
    density = {}
    count_max = 1.0
    for label, analysis in analyses.items():
        matrix = np.asarray(analysis[matrix_key], dtype=float)
        clipped = np.clip(matrix, y_edges[0], np.nextafter(y_edges[-1], y_edges[0]))
        layers = np.broadcast_to(
            np.arange(1, matrix.shape[1] + 1, dtype=float),
            matrix.shape,
        )
        counts, _, _ = np.histogram2d(
            layers.reshape(-1),
            clipped.reshape(-1),
            bins=(np.arange(0.5, matrix.shape[1] + 1.5), y_edges),
        )
        density[label] = counts
        count_max = max(count_max, float(counts.max()))
    return y_edges, density, count_max


def _secondary_z_axis(ax, layer_z):
    layer_numbers = np.arange(1, layer_z.size + 1, dtype=float)

    def layer_to_z(values):
        return np.interp(np.asarray(values, dtype=float), layer_numbers, layer_z)

    def z_to_layer(values):
        return np.interp(np.asarray(values, dtype=float), layer_z, layer_numbers)

    secondary = ax.secondary_xaxis("top", functions=(layer_to_z, z_to_layer))
    tick_layers = np.unique(
        np.clip(
            np.asarray([1, 5, 9, 13, 17, 21, 25, 29, layer_z.size]),
            1,
            layer_z.size,
        )
    )
    secondary.set_xticks(layer_z[tick_layers - 1])
    secondary.set_xticklabels([f"{value:.0f}" for value in layer_z[tick_layers - 1]])
    secondary.set_xlabel("physical ECal z [mm]", fontsize=12, labelpad=7)
    secondary.tick_params(labelsize=10)


def plot_longitudinal_densities(analyses, output_stem):
    specs = (
        (
            "layer_hits",
            "ECal hits in event layer",
            "event-layer hit multiplicity",
        ),
        (
            "layer_energy",
            "reconstructed energy in event layer [MeV]",
            "event-layer reconstructed energy",
        ),
    )
    prepared = {
        matrix_key: _layer_density(analyses, matrix_key)
        for matrix_key, _ylabel, _title in specs
    }
    count_max = max(values[2] for values in prepared.values())
    fig, axes = plt.subplots(
        len(analyses),
        len(specs),
        figsize=(16.0, 5.4 * len(analyses)),
        squeeze=False,
        layout="constrained",
    )
    mesh = None
    for row_index, (label, analysis) in enumerate(analyses.items()):
        layer_z = analysis["layer_z"]
        layers = np.arange(1, layer_z.size + 1, dtype=float)
        x_edges = np.arange(0.5, layer_z.size + 1.5)
        for column_index, (matrix_key, ylabel, title) in enumerate(specs):
            ax = axes[row_index, column_index]
            y_edges, density, _metric_max = prepared[matrix_key]
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                _populated(density[label].T),
                cmap="viridis",
                norm=LogNorm(vmin=1.0, vmax=count_max),
                shading="flat",
            )
            mean, low, high = _mean_ci(analysis[matrix_key])
            _draw_mean_curve(ax, layers, mean, low, high)
            ax.set_xlim(0.5, layer_z.size + 0.5)
            ax.set_ylim(max(0.0, y_edges[0]), y_edges[-1])
            ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
            ax.set_xlabel("ECal layer")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label}: {title}")
            ax.legend(loc="upper right")
            ax.grid(False)
            if row_index == 0:
                _secondary_z_axis(ax, layer_z)
    fig.colorbar(
        mesh,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
        label="event-layer observations per populated rectangular bin",
    )
    fig.suptitle(
        "Event-by-event longitudinal distributions with mean profiles",
        fontsize=21,
    )
    fig.text(
        0.5,
        -0.005,
        "For each metric, the final y bin includes the upper 0.5% tail.",
        ha="center",
        fontsize=10,
    )
    return _save_figure(fig, output_stem)


def _centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    midpoint = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        [
            [centers[0] - 0.5 * (centers[1] - centers[0])],
            midpoint,
            [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
        ]
    )


def _shared_y_slice(analyses, lower_quantile=0.001, upper_quantile=0.999):
    pooled = np.sum(
        [analysis["yz_hit_counts"].sum(axis=0) for analysis in analyses.values()],
        axis=0,
    )
    cumulative = np.cumsum(pooled)
    if cumulative[-1] <= 0:
        return slice(0, pooled.size), BASE_Y_EDGES
    low_index = int(np.searchsorted(cumulative, lower_quantile * cumulative[-1]))
    high_index = int(np.searchsorted(cumulative, upper_quantile * cumulative[-1]))
    low_value = BASE_Y_EDGES[max(0, low_index)]
    high_value = BASE_Y_EDGES[min(pooled.size, high_index + 1)]
    low_value = max(BASE_Y_EDGES[0], 25.0 * math.floor(low_value / 25.0))
    high_value = min(BASE_Y_EDGES[-1], 25.0 * math.ceil(high_value / 25.0))
    start = int(np.searchsorted(BASE_Y_EDGES, low_value, side="left"))
    stop_edge = int(np.searchsorted(BASE_Y_EDGES, high_value, side="right")) - 1
    stop_bin = max(start + 1, min(pooled.size, stop_edge))
    return slice(start, stop_bin), BASE_Y_EDGES[start : stop_bin + 1]


def _draw_center_curve(ax, z, y, label):
    ax.plot(z, y, color="#111827", linewidth=4.0, zorder=4)
    return ax.plot(
        z,
        y,
        color="white",
        linewidth=2.0,
        label=label,
        zorder=5,
    )[0]


def plot_aggregate_yz_context(analyses, output_stem):
    y_slice, y_edges = _shared_y_slice(analyses)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    hit_vmax = max(
        float(analysis["yz_hit_counts"][:, y_slice].max())
        for analysis in analyses.values()
    )
    energy_parts = [
        analysis["yz_energy"][:, y_slice]
        for analysis in analyses.values()
    ]
    energy_positive = np.concatenate(
        [part[part > 0.0] for part in energy_parts]
    )
    energy_vmin = max(1e-9, float(energy_positive.min()))
    energy_vmax = float(energy_positive.max())
    fig, axes = plt.subplots(
        len(analyses),
        3,
        figsize=(18.0, 5.4 * len(analyses)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.2, 4.0, 4.0]},
        layout="constrained",
    )
    hit_mesh = None
    energy_mesh = None
    for row_index, (label, analysis) in enumerate(analyses.items()):
        z = analysis["layer_z"]
        z_edges = _centers_to_edges(z)
        tpad_counts = analysis["tpad_y_counts"][y_slice].astype(float)
        tpad_fraction = tpad_counts / max(1.0, float(tpad_counts.sum()))
        marginal = axes[row_index, 0]
        marginal.fill_betweenx(
            y_centers,
            0.0,
            tpad_fraction,
            color="#6b7280",
            alpha=0.45,
            step="mid",
        )
        marginal.plot(tpad_fraction, y_centers, color="#111827", linewidth=1.7)
        marginal.invert_xaxis()
        marginal.xaxis.set_major_formatter(PercentFormatter(1.0))
        marginal.set_xlabel("fraction of\nTPad tokens per bin", fontsize=12)
        marginal.set_ylabel("transverse y [mm]")
        marginal.set_title(f"{label}: TPad y")
        marginal.grid(True, axis="x", alpha=0.25)

        hit_ax = axes[row_index, 1]
        hit_mesh = hit_ax.pcolormesh(
            z_edges,
            y_edges,
            _populated(analysis["yz_hit_counts"][:, y_slice].T),
            cmap="viridis",
            norm=LogNorm(vmin=1.0, vmax=hit_vmax),
            shading="flat",
        )
        _draw_center_curve(
            hit_ax,
            z,
            np.where(
                analysis["layer_hit_count"]
                >= max(100.0, 0.01 * float(analysis["layer_hit_count"].max())),
                analysis["layer_mean_hit_y"],
                np.nan,
            ),
            "mean hit y",
        )
        hit_ax.set_xlabel("physical ECal z [mm]")
        hit_ax.set_ylabel("ECal hit y [mm]")
        hit_ax.set_title(f"{label}: ECal hit-occurrence density")
        hit_ax.legend(loc="upper right")

        energy_ax = axes[row_index, 2]
        energy_mesh = energy_ax.pcolormesh(
            z_edges,
            y_edges,
            np.ma.masked_less_equal(analysis["yz_energy"][:, y_slice].T, 0.0),
            cmap="viridis",
            norm=LogNorm(vmin=energy_vmin, vmax=energy_vmax),
            shading="flat",
        )
        _draw_center_curve(
            energy_ax,
            z,
            np.where(
                analysis["layer_energy_sum"]
                >= 0.01 * float(analysis["layer_energy_sum"].max()),
                analysis["layer_mean_energy_y"],
                np.nan,
            ),
            "energy-weighted mean y",
        )
        energy_ax.set_xlabel("physical ECal z [mm]")
        energy_ax.set_ylabel("ECal hit y [mm]")
        energy_ax.set_title(f"{label}: reconstructed-energy density")
        energy_ax.legend(loc="upper right")
        for ax in axes[row_index]:
            ax.set_ylim(y_edges[0], y_edges[-1])
            ax.tick_params(labelsize=12)
    fig.colorbar(
        hit_mesh,
        ax=axes[:, 1].ravel().tolist(),
        fraction=0.035,
        pad=0.02,
        label="hit occurrences per populated rectangular bin",
    )
    fig.colorbar(
        energy_mesh,
        ax=axes[:, 2].ravel().tolist(),
        fraction=0.035,
        pad=0.02,
        label="summed reconstructed energy per populated bin [MeV]",
    )
    fig.suptitle(
        "Aggregate longitudinal detector view with TPad-y context",
        fontsize=21,
    )
    return _save_figure(fig, output_stem)


def _tpad_contingency(analyses):
    true_values = sorted(
        {
            int(value)
            for analysis in analyses.values()
            for value in analysis["electron_counts"]
        }
    )
    observed_values = sorted(
        {
            int(value)
            for analysis in analyses.values()
            for value in analysis["tpad_counts"]
        }
    )
    matrix = np.zeros((len(true_values), len(observed_values)), dtype=np.int64)
    true_to_row = {value: index for index, value in enumerate(true_values)}
    observed_to_column = {value: index for index, value in enumerate(observed_values)}
    for analysis in analyses.values():
        for true_count, observed_count in zip(
            analysis["electron_counts"],
            analysis["tpad_counts"],
        ):
            matrix[
                true_to_row[int(true_count)],
                observed_to_column[int(observed_count)],
            ] += 1
    row_total = matrix.sum(axis=1, keepdims=True)
    fraction = np.divide(
        matrix,
        row_total,
        out=np.zeros_like(matrix, dtype=float),
        where=row_total > 0,
    )
    return true_values, observed_values, matrix, fraction


def plot_tpad_count_matrix(analyses, output_stem):
    true_values, observed_values, matrix, fraction = _tpad_contingency(analyses)
    fig, ax = plt.subplots(figsize=(9.2, 5.7), layout="constrained")
    image = ax.imshow(
        fraction,
        cmap="Blues",
        norm=Normalize(vmin=0.0, vmax=1.0),
        aspect="auto",
    )
    for row_index, true_count in enumerate(true_values):
        for column_index, observed_count in enumerate(observed_values):
            value = fraction[row_index, column_index]
            color = "white" if value >= 0.52 else "#111827"
            percentage = 100.0 * value
            percentage_text = (
                "<0.1%"
                if matrix[row_index, column_index] > 0 and percentage < 0.05
                else f"{percentage:.1f}%"
            )
            ax.text(
                column_index,
                row_index,
                f"{percentage_text}\n(n={matrix[row_index, column_index]:,})",
                ha="center",
                va="center",
                color=color,
                fontsize=13,
            )
        if true_count in observed_values:
            column_index = observed_values.index(true_count)
            ax.add_patch(
                Rectangle(
                    (column_index - 0.49, row_index - 0.49),
                    0.98,
                    0.98,
                    fill=False,
                    edgecolor="#dc2626",
                    linewidth=2.4,
                )
            )
    ax.set_xticks(np.arange(len(observed_values)), observed_values)
    ax.set_yticks(np.arange(len(true_values)), true_values)
    ax.set_xlabel("observed number of TPad track tokens")
    ax.set_ylabel("true generated electron multiplicity")
    ax.set_title("TPad token multiplicity as an electron-count proxy")
    fig.colorbar(image, ax=ax, label="percentage within true multiplicity")
    return _save_figure(fig, output_stem)


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


def _profile_rows(analyses):
    rows = []
    for label, analysis in analyses.items():
        hit_mean, hit_low, hit_high = _mean_ci(analysis["layer_hits"])
        energy_mean, energy_low, energy_high = _mean_ci(analysis["layer_energy"])
        for layer_index, z_value in enumerate(analysis["layer_z"]):
            rows.append(
                {
                    "sample": label,
                    "layer": layer_index + 1,
                    "z_mm": float(z_value),
                    "mean_hits_per_event": float(hit_mean[layer_index]),
                    "hit_mean_ci95_low": float(hit_low[layer_index]),
                    "hit_mean_ci95_high": float(hit_high[layer_index]),
                    "mean_energy_per_event_mev": float(energy_mean[layer_index]),
                    "energy_mean_ci95_low": float(energy_low[layer_index]),
                    "energy_mean_ci95_high": float(energy_high[layer_index]),
                    "mean_hit_y_mm": float(analysis["layer_mean_hit_y"][layer_index]),
                    "energy_weighted_mean_y_mm": float(
                        analysis["layer_mean_energy_y"][layer_index]
                    ),
                }
            )
    return rows


def _tpad_rows(analyses):
    true_values, observed_values, matrix, fraction = _tpad_contingency(analyses)
    rows = []
    for row_index, true_count in enumerate(true_values):
        for column_index, observed_count in enumerate(observed_values):
            rows.append(
                {
                    "true_electron_count": true_count,
                    "observed_tpad_tokens": observed_count,
                    "event_count": int(matrix[row_index, column_index]),
                    "row_percentage": float(100.0 * fraction[row_index, column_index]),
                }
            )
    return rows


def _build_summary(analyses):
    true_values, observed_values, matrix, fraction = _tpad_contingency(analyses)
    per_true = {}
    for row_index, true_count in enumerate(true_values):
        diagonal = (
            fraction[row_index, observed_values.index(true_count)]
            if true_count in observed_values
            else 0.0
        )
        deficit = sum(
            fraction[row_index, column_index]
            for column_index, value in enumerate(observed_values)
            if value < true_count
        )
        excess = sum(
            fraction[row_index, column_index]
            for column_index, value in enumerate(observed_values)
            if value > true_count
        )
        per_true[str(true_count)] = {
            "num_events": int(matrix[row_index].sum()),
            "exact_token_count_fraction": float(diagonal),
            "token_deficit_fraction": float(deficit),
            "token_excess_fraction": float(excess),
            "by_observed_token_count": {
                str(observed_count): {
                    "count": int(matrix[row_index, column_index]),
                    "fraction": float(fraction[row_index, column_index]),
                }
                for column_index, observed_count in enumerate(observed_values)
            },
        }
    return {
        "scope": "model-independent deterministic 20k cache-prefix detector context",
        "samples": {
            label: {
                "cache_dir": analysis["cache_dir"],
                "num_events": analysis["num_events"],
                "num_layers": int(analysis["layer_z"].size),
                "range_checks": analysis["range_checks"],
            }
            for label, analysis in analyses.items()
        },
        "tpad_token_multiplicity": {
            "interpretation": (
                "Observed TPad track count is treated as a reconstruction-level electron-count proxy, not a learned classifier prediction."
            ),
            "per_true_electron_count": per_true,
        },
    }


def _write_readme(output_dir, analyses, summary):
    lines = [
        "# Detector-context density development plots for the 20k samples",
        "",
        "This bundle generalizes the validation-only detector-context plots to the",
        "same deterministic first 20,000 tensor-cache events per 2e and 3e sample.",
        "It is model-independent: no checkpoint, inference output, or split membership",
        "is used.",
        "",
        "## Plot index",
        "",
        "1. `01_longitudinal_event_layer_densities` replaces mean-only longitudinal",
        "   curves with rectangular-bin event-layer densities. Zero-count bins are",
        "   white, the logarithmic count scale begins at one occurrence, and the event",
        "   mean with its 95% confidence interval is overlaid on each heatmap.",
        "2. `02_aggregate_yz_tpad_context` generalizes the single-event y-z view.",
        "   Each row shows the aggregate TPad y-centroid marginal, ECal hit-occurrence",
        "   density in y-z, and reconstructed-energy density in y-z for one sample.",
        "   The rectangular bins are 5 mm high in y.",
        "3. `03_tpad_token_multiplicity_matrix` replaces raw-count bars with a",
        "   row-normalized contingency matrix. Every cell gives both the percentage",
        "   within the true electron multiplicity and the underlying event count.",
        "",
        "## Why the TPad is a marginal in the aggregate y-z view",
        "",
        "The stored TPad token has a one-dimensional y-like centroid but no stored",
        "physical z coordinate. In a single event it can be drawn just upstream of the",
        "ECal as a schematic marker. Across all events, assigning every TPad centroid an",
        "invented z value would imply detector information that is not present. The shared",
        "y marginal preserves the useful comparison honestly: its vertical coordinate is",
        "directly comparable with the ECal y axis, while physical z is reserved for ECal",
        "hits. The black/white line on each ECal density gives the mean y at each layer.",
        "The horizontal banding in the ECal heatmaps reflects the discrete y coordinates",
        "of the detector-cell centers; it is detector geometry rather than missing data.",
        "",
        "## Interpreting the TPad count matrix",
        "",
        "The observed number of TPad track tokens can be interpreted as a simple",
        "reconstruction-level proxy for electron multiplicity, so a confusion-matrix-like",
        "display is useful. It is more precisely a contingency matrix: no classifier was",
        "trained to predict the count. Red outlines mark the expected diagonal where the",
        "TPad token count equals the generated electron count.",
        "",
    ]
    for true_count, values in summary["tpad_token_multiplicity"][
        "per_true_electron_count"
    ].items():
        excess_percentage = 100.0 * values["token_excess_fraction"]
        excess_text = (
            "<0.1%"
            if 0.0 < excess_percentage < 0.05
            else f"{excess_percentage:.1f}%"
        )
        lines.append(
            f"- True {true_count}e: exact token count "
            f"{100.0 * values['exact_token_count_fraction']:.1f}%, token deficit "
            f"{100.0 * values['token_deficit_fraction']:.1f}%, token excess "
            f"{excess_text}."
        )
    lines.extend(
        [
            "",
            "The numerical profile and contingency tables are in `data/`; compact array",
            "data sufficient to reproduce the heatmaps is stored in `data/density_arrays.npz`.",
            "",
        ]
    )
    path = Path(output_dir) / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _save_density_arrays(path, analyses):
    arrays = {"base_y_edges_mm": BASE_Y_EDGES}
    for label, analysis in analyses.items():
        prefix = label.replace("-", "_")
        for key in (
            "layer_z",
            "layer_hits",
            "layer_energy",
            "yz_hit_counts",
            "yz_energy",
            "layer_mean_hit_y",
            "layer_mean_energy_y",
            "layer_hit_count",
            "layer_energy_sum",
            "tpad_y_counts",
            "tpad_counts",
            "electron_counts",
        ):
            arrays[f"{prefix}_{key}"] = analysis[key]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
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
    for label, cache_dir in sample_specs:
        analyses[label] = _scan_sample(
            label,
            cache_dir,
            max_events=args.max_events,
            progress_every=args.progress_every,
        )
    reference_z = None
    for analysis in analyses.values():
        layer_z = analysis["layer_z"]
        if reference_z is None:
            reference_z = layer_z
        elif reference_z.shape != layer_z.shape or not np.allclose(reference_z, layer_z):
            raise ValueError("Samples do not share the same ECal layer-z coordinates.")
        if any(analysis["range_checks"].values()):
            raise ValueError(
                f"Configured y range did not contain all {analysis['label']} values: "
                f"{analysis['range_checks']}"
            )

    generated = []
    generated.extend(
        plot_longitudinal_densities(
            analyses,
            output_dir / "01_longitudinal_event_layer_densities",
        )
    )
    generated.extend(
        plot_aggregate_yz_context(
            analyses,
            output_dir / "02_aggregate_yz_tpad_context",
        )
    )
    generated.extend(
        plot_tpad_count_matrix(
            analyses,
            output_dir / "03_tpad_token_multiplicity_matrix",
        )
    )
    generated.extend(
        [
            _write_csv(
                output_dir / "data/longitudinal_profiles.csv",
                _profile_rows(analyses),
            ),
            _write_csv(
                output_dir / "data/tpad_token_multiplicity.csv",
                _tpad_rows(analyses),
            ),
            _save_density_arrays(
                output_dir / "data/density_arrays.npz",
                analyses,
            ),
        ]
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
                    "num_layers": int(analysis["layer_z"].size),
                }
                for label, analysis in analyses.items()
            },
            "generated_files": sorted(
                [str(Path(item).resolve().relative_to(output_dir)) for item in generated]
                + [manifest_path.name]
            ),
        },
    )
    print(f"Saved {len(generated) + 1} artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
