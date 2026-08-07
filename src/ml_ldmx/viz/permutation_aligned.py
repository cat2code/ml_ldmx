"""Report-facing plots for permutation-aligned hit-grouping evaluation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


MODEL_COLORS = {
    "2e": "#2563eb",
    "3e": "#d97706",
}
DEPTH_RANGES = ((1, 20), (21, 32))

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 19,
    }
)


def _count_norm(max_count):
    """Scale populated count bins logarithmically from one occurrence."""
    return LogNorm(vmin=1.0, vmax=max(1.0, float(max_count)))


def _populated_counts(counts):
    """Mask empty histogram bins so the axes background remains white."""
    return np.ma.masked_less(np.asarray(counts, dtype=float), 1.0)


def _draw_mean_errorbars(ax, x, mean, low, high, label="event mean ± 95% CI"):
    """Draw readable bootstrap intervals over a density heatmap."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    yerr = np.vstack(
        [
            np.clip(mean - low, 0.0, None),
            np.clip(high - mean, 0.0, None),
        ]
    )
    ax.errorbar(
        x,
        mean,
        yerr=yerr,
        fmt="-o",
        color="#111827",
        ecolor="#111827",
        linewidth=4.0,
        elinewidth=3.6,
        capsize=5,
        markersize=6.5,
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
        linewidth=2.1,
        elinewidth=1.8,
        capsize=3.5,
        markersize=5.2,
        label=label,
        zorder=5,
    )


def save_figure(fig, output_stem, dpi=300):
    """Save review-friendly PNG and vector PDF versions of a figure."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _human_count(value):
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _draw_confusion(ax, matrix, title, annotate_counts=True):
    matrix = np.asarray(matrix, dtype=np.int64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues", origin="upper")
    labels = [str(index + 1) for index in range(matrix.shape[0])]
    ax.set_xticks(range(matrix.shape[1]), labels=labels)
    ax.set_yticks(range(matrix.shape[0]), labels=labels)
    ax.set_xlabel("predicted group")
    ax.set_ylabel("true group")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            fraction = normalized[row, column]
            label = f"{100.0 * fraction:.1f}%"
            if annotate_counts:
                label += f"\n({_human_count(matrix[row, column])})"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=12,
                color="white" if fraction > 0.52 else "#111827",
            )
    return image


def plot_event_accuracy_distributions(analyses, output_stem):
    fig, ax = plt.subplots(figsize=(11.5, 6.2), layout="constrained")
    bins = np.linspace(0.0, 1.0, 101)
    for label, analysis in analyses.items():
        color = MODEL_COLORS.get(label, "#4c78a8")
        records = analyses[label]["event_records"]
        values = np.asarray(
            [record["aligned_event_accuracy"] for record in records],
            dtype=float,
        )
        median = float(np.median(values))
        ax.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.42,
            edgecolor=color,
            linewidth=0.8,
            label=f"{label} events (N={values.size:,})",
        )
        ax.axvline(
            median,
            color=color,
            linewidth=2.5,
            linestyle="--",
            label=f"{label} median={median:.3f}",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("event hit accuracy")
    ax.set_ylabel("events per accuracy bin")
    ax.set_title("Event accuracy distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncols=2)
    return save_figure(fig, output_stem)


def plot_event_accuracy_distributions_with_boxplots(analyses, output_stem):
    """Show event histograms with a compact shared-axis box-plot summary."""
    fig, (hist_ax, box_ax) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": (5.0, 1.05), "hspace": 0.04},
        layout="constrained",
    )
    bins = np.linspace(0.0, 1.0, 101)
    values_by_label = {}
    for label, analysis in analyses.items():
        color = MODEL_COLORS.get(label, "#4c78a8")
        values = np.asarray(
            [record["aligned_event_accuracy"] for record in analysis["event_records"]],
            dtype=float,
        )
        values_by_label[label] = values
        hist_ax.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=0.8,
            label=f"{label} events (N={values.size:,})",
        )
        hist_ax.axvline(
            float(np.median(values)),
            color=color,
            linewidth=2.4,
            linestyle="--",
            label=f"{label} median={np.median(values):.3f}",
        )

    labels = list(values_by_label)
    positions = np.arange(len(labels), 0, -1)
    boxplot = box_ax.boxplot(
        [values_by_label[label] for label in labels],
        vert=False,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "#111827", "linewidth": 2.0},
        whiskerprops={"color": "#374151", "linewidth": 1.5},
        capprops={"color": "#374151", "linewidth": 1.5},
    )
    for patch, label in zip(boxplot["boxes"], labels):
        patch.set_facecolor(MODEL_COLORS.get(label, "#4c78a8"))
        patch.set_edgecolor(MODEL_COLORS.get(label, "#4c78a8"))
        patch.set_alpha(0.42)
    box_ax.set_yticks(positions, labels=labels)
    box_ax.set_ylim(0.45, len(labels) + 0.55)
    box_ax.set_xlim(0.0, 1.0)
    box_ax.set_xlabel("event hit accuracy")
    box_ax.grid(axis="x", alpha=0.2)

    hist_ax.set_ylabel("events per accuracy bin")
    hist_ax.set_title("Event accuracy distribution with box-plot summary")
    hist_ax.grid(axis="y", alpha=0.25)
    hist_ax.legend(loc="upper left", ncols=2)
    return save_figure(fig, output_stem)


def plot_overall_confusions(analyses, output_stem):
    labels = list(analyses)
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(5.8 * len(labels), 5.0),
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    image = None
    for ax, label in zip(axes, labels):
        analysis = analyses[label]
        summary = analysis["summary"]
        title = (
            f"{label} validation\n"
            f"pooled hit accuracy={summary['pooled_aligned_hit_accuracy']:.3f}"
        )
        image = _draw_confusion(ax, analysis["confusion"], title)
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.tolist(),
            fraction=0.035,
            pad=0.02,
            shrink=0.86,
            label="row-normalized hit fraction",
        )
    fig.suptitle("Hit confusion matrices")
    return save_figure(fig, output_stem)


def plot_depth_range_confusions(analyses, output_stem):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 10.5),
        layout="constrained",
    )
    image = None
    for row_index, (label, analysis) in enumerate(analyses.items()):
        for column_index, (layer_start, layer_stop) in enumerate(DEPTH_RANGES):
            matrix = analysis["depth_confusions"][
                f"{layer_start:02d}_{layer_stop:02d}"
            ]
            accuracy = np.trace(matrix) / max(1, matrix.sum())
            image = _draw_confusion(
                axes[row_index, column_index],
                matrix,
                (
                    f"{label}, layers {layer_start}–{layer_stop}\n"
                    f"pooled hit accuracy={accuracy:.3f}"
                ),
                annotate_counts=False,
            )
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.86,
            label="row-normalized hit fraction",
        )
    fig.suptitle("Hit confusion matrices by ECal depth")
    return save_figure(fig, output_stem)


def plot_all_confusions(analyses, output_stem):
    """Combine overall and two-range depth confusion matrices in one figure."""
    labels = list(analyses)
    if len(labels) != 2:
        raise ValueError("The combined confusion layout requires exactly two samples.")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.8, 15.0),
        layout="constrained",
    )
    image = None
    for column_index, label in enumerate(labels):
        analysis = analyses[label]
        summary = analysis["summary"]
        image = _draw_confusion(
            axes[0, column_index],
            analysis["confusion"],
            (
                f"{label}, all ECal layers\n"
                f"pooled hit accuracy={summary['pooled_aligned_hit_accuracy']:.3f}"
            ),
            annotate_counts=True,
        )

    for column_index, label in enumerate(labels):
        analysis = analyses[label]
        for row_index, (layer_start, layer_stop) in enumerate(
            DEPTH_RANGES,
            start=1,
        ):
            matrix = analysis["depth_confusions"][
                f"{layer_start:02d}_{layer_stop:02d}"
            ]
            accuracy = np.trace(matrix) / max(1, matrix.sum())
            image = _draw_confusion(
                axes[row_index, column_index],
                matrix,
                (
                    f"{label}, layers {layer_start}–{layer_stop}\n"
                    f"pooled hit accuracy={accuracy:.3f}"
                ),
                annotate_counts=False,
            )

    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
            shrink=0.88,
            label="row-normalized hit fraction",
        )
    fig.suptitle("Hit confusion matrices")
    return save_figure(fig, output_stem)


def _profile_arrays(profile, metric):
    layers = np.asarray([row["layer"] for row in profile], dtype=float)
    mean = np.asarray([row[f"{metric}_mean"] for row in profile], dtype=float)
    low = np.asarray([row[f"{metric}_ci_low"] for row in profile], dtype=float)
    high = np.asarray([row[f"{metric}_ci_high"] for row in profile], dtype=float)
    return layers, mean, low, high


def plot_accuracy_by_layer(analyses, output_stem):
    specs = (
        (
            "aligned_event_layer_accuracy",
            "event-balanced hit accuracy",
            "Hit-count accuracy",
        ),
        (
            "aligned_event_layer_energy_weighted_accuracy",
            "event-balanced energy-weighted accuracy",
            "Energy-weighted accuracy",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, (metric, ylabel, title) in zip(axes, specs):
        for label, analysis in analyses.items():
            layer, mean, low, high = _profile_arrays(analysis["layer_profiles"], metric)
            color = MODEL_COLORS.get(label)
            ax.plot(layer, mean, marker="o", markersize=3.2, linewidth=1.7, color=color, label=label)
            ax.fill_between(layer, low, high, color=color, alpha=0.16)
        ax.set_xlim(1, 32)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        ax.set_xlabel("ECal layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(title="sample")
    fig.suptitle("Permutation-aligned accuracy along the ECal depth")
    fig.tight_layout()
    return save_figure(fig, output_stem)


def plot_layer_accuracy_distribution(
    analysis,
    label,
    output_stem,
    metric="hit",
):
    if metric == "hit":
        row_key = "aligned_event_layer_accuracy"
        profile_key = "aligned_event_layer_accuracy"
        ylabel = "event hit accuracy within layer"
        title_metric = "hit accuracy"
    elif metric == "energy":
        row_key = "aligned_event_layer_energy_weighted_accuracy"
        profile_key = "aligned_event_layer_energy_weighted_accuracy"
        ylabel = "event energy-weighted accuracy within layer"
        title_metric = "energy-weighted accuracy"
    else:
        raise ValueError(f"Unknown layer-accuracy metric {metric!r}.")

    rows = analysis["layer_rows"]
    finite_rows = [
        row
        for row in rows
        if row.get(row_key) is not None and np.isfinite(float(row[row_key]))
    ]
    layer = np.asarray([row["layer"] for row in finite_rows], dtype=float)
    accuracy = np.asarray([row[row_key] for row in finite_rows], dtype=float)
    x_edges = np.arange(0.5, 33.5, 1.0)
    y_edges = np.linspace(0.0, 1.0, 41)
    counts, _, _ = np.histogram2d(layer, accuracy, bins=(x_edges, y_edges))

    fig, ax = plt.subplots(figsize=(12.5, 6.2), layout="constrained")
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        _populated_counts(counts.T),
        cmap="viridis",
        norm=_count_norm(counts.max()),
        shading="flat",
    )
    profile = analysis["layer_profiles"]
    profile_layer, mean, low, high = _profile_arrays(
        profile,
        profile_key,
    )
    _draw_mean_errorbars(
        ax,
        profile_layer,
        mean,
        low,
        high,
    )
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.set_xlabel("ECal layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{label} event-layer {title_metric} distribution")
    ax.legend(loc="lower left")
    fig.colorbar(
        mesh,
        ax=ax,
        shrink=0.88,
        label="event-layer observations per bin",
    )
    return save_figure(fig, output_stem)


def _finite_xy(records, x_key, y_key="aligned_event_accuracy"):
    values = [
        (record.get(x_key), record.get(y_key))
        for record in records
        if record.get(x_key) is not None and record.get(y_key) is not None
    ]
    values = [
        (float(x), float(y))
        for x, y in values
        if np.isfinite(float(x)) and np.isfinite(float(y))
    ]
    if not values:
        return np.asarray([]), np.asarray([])
    return np.asarray([value[0] for value in values]), np.asarray([value[1] for value in values])


def _density_edges(
    x,
    x_range=None,
    x_bins=30,
    y_bins=30,
    x_scale="linear",
):
    if x_range is None:
        if x_scale == "log":
            positive = x[x > 0]
            if positive.size == 0:
                raise ValueError("A logarithmic density axis requires positive values.")
            x_min = float(np.min(positive))
            x_max = float(np.max(positive))
        else:
            x_min = float(np.min(x))
            x_max = float(np.max(x))
    else:
        x_min, x_max = map(float, x_range)
    if x_scale == "log" and x_min <= 0:
        raise ValueError("A logarithmic density-axis range must be positive.")
    if x_max <= x_min:
        x_max = x_min * 10.0 if x_scale == "log" else x_min + 1.0
    if x_scale == "log":
        x_edges = np.geomspace(x_min, x_max, int(x_bins) + 1)
    else:
        x_edges = np.linspace(x_min, x_max, int(x_bins) + 1)
    return x_edges, np.linspace(0.0, 1.0, int(y_bins) + 1)


def _draw_density_profile(
    ax,
    records,
    profile,
    x_key,
    xlabel,
    title,
    x_reference=None,
    x_range=None,
    count_vmax=None,
    x_scale="linear",
    clip_to_range=False,
):
    x, y = _finite_xy(records, x_key)
    if x.size == 0:
        ax.axis("off")
        return None
    x_edges, y_edges = _density_edges(
        x,
        x_range=x_range,
        x_scale=x_scale,
    )
    valid = x > 0 if x_scale == "log" else np.ones(x.shape, dtype=bool)
    underflow = int(np.count_nonzero(valid & (x < x_edges[0])))
    overflow = int(np.count_nonzero(valid & (x > x_edges[-1])))
    if clip_to_range:
        x_for_hist = np.clip(
            x[valid],
            x_edges[0],
            np.nextafter(x_edges[-1], x_edges[0]),
        )
        y_for_hist = y[valid]
    else:
        visible = valid & (x >= x_edges[0]) & (x <= x_edges[-1])
        x_for_hist = x[visible]
        y_for_hist = y[visible]
    counts, _, _ = np.histogram2d(
        x_for_hist,
        y_for_hist,
        bins=(x_edges, y_edges),
    )
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        _populated_counts(counts.T),
        cmap="viridis",
        norm=_count_norm(
            float(counts.max()) if count_vmax is None else float(count_vmax)
        ),
        shading="flat",
    )
    if profile:
        x_mean = np.asarray([row["x_mean"] for row in profile], dtype=float)
        y_mean = np.asarray([row["y_mean"] for row in profile], dtype=float)
        y_low = np.asarray([row["y_ci_low"] for row in profile], dtype=float)
        y_high = np.asarray([row["y_ci_high"] for row in profile], dtype=float)
        in_axis = (x_mean >= x_edges[0]) & (x_mean <= x_edges[-1])
        if x_scale == "log":
            in_axis &= x_mean > 0
        _draw_mean_errorbars(
            ax,
            x_mean[in_axis],
            y_mean[in_axis],
            y_low[in_axis],
            y_high[in_axis],
        )
    if x_reference is not None and x_edges[0] <= x_reference <= x_edges[-1]:
        ax.axvline(
            float(x_reference),
            color="#ef4444",
            linewidth=1.7,
            linestyle="--",
            label=r"$d_{\min}=R_{\mathrm{M}}$",
        )
        ax.legend(loc="lower right")
    if clip_to_range and (underflow or overflow):
        clipped = underflow + overflow
        ax.text(
            0.02,
            0.98,
            f"{clipped:,} events ({100.0 * clipped / x.size:.1f}%) "
            "clipped to edge bin",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.88,
            },
        )
    ax.set_xscale(x_scale)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("event accuracy")
    ax.set_title(title)
    ax.grid(False)
    return mesh


def _density_count_max(
    records,
    x_key,
    x_range=None,
    x_scale="linear",
    clip_to_range=False,
):
    x, y = _finite_xy(records, x_key)
    if x.size == 0:
        return 1.0
    x_edges, y_edges = _density_edges(
        x,
        x_range=x_range,
        x_scale=x_scale,
    )
    valid = x > 0 if x_scale == "log" else np.ones(x.shape, dtype=bool)
    if clip_to_range:
        x_for_hist = np.clip(
            x[valid],
            x_edges[0],
            np.nextafter(x_edges[-1], x_edges[0]),
        )
        y_for_hist = y[valid]
    else:
        visible = valid & (x >= x_edges[0]) & (x <= x_edges[-1])
        x_for_hist = x[visible]
        y_for_hist = y[visible]
    counts, _, _ = np.histogram2d(
        x_for_hist,
        y_for_hist,
        bins=(x_edges, y_edges),
    )
    return max(1.0, float(counts.max()))


def plot_separation_density(
    analyses,
    output_stem,
    metric_keys,
    xlabel,
    figure_title,
    x_reference=None,
    x_range=None,
    x_scale="linear",
    clip_to_range=False,
):
    labels = list(analyses)
    fig, axes = plt.subplots(
        len(labels),
        len(metric_keys),
        figsize=(7.4 * len(metric_keys) + 1.2, 4.8 * len(labels)),
        squeeze=False,
        layout="constrained",
    )
    common_count_max = 1.0
    for label in labels:
        analysis = analyses[label]
        for x_key, _depth_label in metric_keys:
            common_count_max = max(
                common_count_max,
                _density_count_max(
                    analysis["event_records"],
                    x_key,
                    x_range=x_range,
                    x_scale=x_scale,
                    clip_to_range=clip_to_range,
                ),
            )
    meshes = []
    for row_index, label in enumerate(labels):
        analysis = analyses[label]
        for column_index, (x_key, depth_label) in enumerate(metric_keys):
            profile = analysis["binned_profiles"].get(x_key, [])
            mesh = _draw_density_profile(
                axes[row_index, column_index],
                analysis["event_records"],
                profile,
                x_key,
                xlabel,
                f"{label}, {depth_label}",
                x_reference=x_reference,
                x_range=x_range,
                count_vmax=common_count_max,
                x_scale=x_scale,
                clip_to_range=clip_to_range,
            )
            if mesh is not None:
                meshes.append(mesh)
    if meshes:
        fig.colorbar(
            meshes[-1],
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
            shrink=0.86,
            label="events per rectangular bin",
        )
    fig.suptitle(figure_title)
    return save_figure(fig, output_stem)


def plot_event_confidence_density(analyses, output_stem):
    fig, axes = plt.subplots(
        1,
        len(analyses),
        figsize=(6.3 * len(analyses), 4.8),
        squeeze=False,
        layout="constrained",
    )
    common_count_max = max(
        _density_count_max(analysis["event_records"], "mean_confidence")
        for analysis in analyses.values()
    )
    meshes = []
    for ax, (label, analysis) in zip(axes.flat, analyses.items()):
        mesh = _draw_density_profile(
            ax,
            analysis["event_records"],
            analysis["binned_profiles"].get("mean_confidence", []),
            "mean_confidence",
            "mean max-softmax confidence",
            f"{label} validation events",
            count_vmax=common_count_max,
        )
        if mesh is not None:
            meshes.append(mesh)
    if meshes:
        fig.colorbar(
            meshes[-1],
            ax=axes.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.86,
            label="events per rectangular bin",
        )
    fig.suptitle("Event confidence versus event accuracy")
    return save_figure(fig, output_stem)


def plot_entropy_density(analyses, output_stem):
    fig, axes = plt.subplots(
        len(analyses),
        1,
        figsize=(8.6, 4.7 * len(analyses)),
        squeeze=False,
        layout="constrained",
    )
    common_count_max = max(
        _density_count_max(
            analysis["event_records"],
            "mean_normalized_entropy",
        )
        for analysis in analyses.values()
    )
    meshes = []
    for row_index, (label, analysis) in enumerate(analyses.items()):
        mesh = _draw_density_profile(
            axes[row_index, 0],
            analysis["event_records"],
            analysis["binned_profiles"].get("mean_normalized_entropy", []),
            "mean_normalized_entropy",
            "mean normalized entropy",
            f"{label} validation events",
            count_vmax=common_count_max,
        )
        if mesh is not None:
            meshes.append(mesh)
    if meshes:
        fig.colorbar(
            meshes[-1],
            ax=axes.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.86,
            label="events per rectangular bin",
        )
    fig.suptitle("Mean normalized entropy versus event accuracy")
    return save_figure(fig, output_stem)


def plot_reliability(analyses, output_stem):
    fig, axes = plt.subplots(2, len(analyses), figsize=(6.0 * len(analyses), 7.8), squeeze=False)
    for column, (label, analysis) in enumerate(analyses.items()):
        rows = analysis["calibration_rows"]
        valid = [row for row in rows if row["num_hits"] > 0]
        confidence = np.asarray([row["mean_confidence"] for row in valid], dtype=float)
        accuracy = np.asarray([row["aligned_hit_accuracy"] for row in valid], dtype=float)
        counts = np.asarray([row["num_hits"] for row in rows], dtype=float)
        centers = np.asarray(
            [
                0.5 * (row["confidence_low"] + row["confidence_high"])
                for row in rows
            ],
            dtype=float,
        )

        ax = axes[0, column]
        ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1.2, label="perfect calibration")
        ax.plot(confidence, accuracy, color=MODEL_COLORS.get(label), marker="o", linewidth=2.0, label=label)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("mean confidence in bin")
        ax.set_ylabel("permutation-aligned hit accuracy")
        ax.set_title(f"{label} reliability, ECE={analysis['ece']:.3f}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        ax = axes[1, column]
        ax.bar(centers, counts / max(1.0, counts.sum()), width=0.09, color=MODEL_COLORS.get(label), alpha=0.82)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("max-softmax confidence")
        ax.set_ylabel("fraction of validation hits")
        ax.set_title("Confidence-bin occupancy")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Confidence calibration after whole-event label alignment")
    fig.tight_layout()
    return save_figure(fig, output_stem)


def plot_accuracy_coverage(analyses, output_stem):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 10.0),
        layout="constrained",
    )
    for label, analysis in analyses.items():
        color = MODEL_COLORS.get(label)
        hit_rows = analysis["hit_coverage_rows"]
        hit_coverage = np.asarray([row["hit_coverage"] for row in hit_rows], dtype=float)
        hit_accuracy = np.asarray([row["aligned_hit_accuracy"] for row in hit_rows], dtype=float)
        energy_coverage = np.asarray([row["energy_coverage"] for row in hit_rows], dtype=float)
        axes[0].plot(
            hit_coverage,
            hit_accuracy,
            color=color,
            marker="o",
            linewidth=2.2,
            label=f"{label} hit accuracy",
        )
        axes[0].plot(
            hit_coverage,
            energy_coverage,
            color=color,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            alpha=0.72,
            label=f"{label} retained energy",
        )

        event_rows = analysis["event_coverage_rows"]
        event_coverage = np.asarray([row["event_coverage"] for row in event_rows], dtype=float)
        event_accuracy = np.asarray(
            [row["macro_aligned_event_accuracy"] for row in event_rows],
            dtype=float,
        )
        axes[1].plot(
            event_coverage,
            event_accuracy,
            color=color,
            marker="o",
            linewidth=2.2,
            label=label,
        )

    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xlabel("retained fraction of hits")
    axes[0].set_ylabel("fraction")
    axes[0].set_title("Removing low-confidence hits")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncols=2)

    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("retained fraction of events")
    axes[1].set_ylabel("mean event accuracy")
    axes[1].set_title("Removing low-confidence events")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(title="sample")
    fig.suptitle("Accuracy–coverage trade-off from confidence selection")
    return save_figure(fig, output_stem)


def plot_confidence_by_layer(analyses, output_stem):
    fig, axes = plt.subplots(
        len(analyses),
        1,
        figsize=(11.0, 4.2 * len(analyses)),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, (label, analysis) in zip(axes.flat, analyses.items()):
        profile = analysis["layer_profiles"]
        layer, accuracy, accuracy_low, accuracy_high = _profile_arrays(
            profile,
            "aligned_event_layer_accuracy",
        )
        _, confidence, confidence_low, confidence_high = _profile_arrays(
            profile,
            "mean_confidence",
        )
        color = MODEL_COLORS.get(label)
        ax.plot(layer, accuracy, color=color, marker="o", markersize=4, label="accuracy")
        ax.fill_between(layer, accuracy_low, accuracy_high, color=color, alpha=0.14)
        ax.plot(
            layer,
            confidence,
            color="#0f766e",
            marker="s",
            markersize=3,
            linestyle="--",
            label="mean confidence",
        )
        ax.fill_between(layer, confidence_low, confidence_high, color="#0f766e", alpha=0.12)
        ax.set_xlim(1, 32)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("event-balanced mean")
        ax.set_title(f"{label} validation events")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1, 0].set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    axes[-1, 0].set_xlabel("ECal layer")
    fig.suptitle("Prediction confidence and accuracy along the ECal depth")
    return save_figure(fig, output_stem)
