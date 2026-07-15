import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ml_ldmx.eval.run_comparison import mean_confidence_interval, quantile_edges


def plot_history(history, run_dir, title_prefix="ECAL/TPAD MLPF-lite"):
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    ax.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{title_prefix} loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_history.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    ax.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title_prefix} accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy_history.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(confusion, valid_labels, output_path, title):
    confusion = torch.as_tensor(confusion, dtype=torch.float64)
    row_sums = confusion.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized = confusion / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalized.numpy(), vmin=0, vmax=1, cmap="Blues", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_xticks(range(len(valid_labels)), labels=[str(label) for label in valid_labels])
    ax.set_yticks(range(len(valid_labels)), labels=[str(label) for label in valid_labels])

    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            count = int(confusion[row, col].item())
            frac = normalized[row, col].item()
            text_color = "white" if frac > 0.5 else "black"
            ax.text(col, row, f"{count}\n{frac:.2f}", ha="center", va="center", color=text_color)

    fig.colorbar(image, ax=ax, label="row-normalized fraction")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig, ax


def plot_event_accuracy_overview(records, output_path, title, annotate_worst=8):
    """Plot per-event hit accuracy across one validation/test split."""
    if not records:
        return

    records = sorted(records, key=lambda record: record.get("split_position", record["event_idx"]))
    positions = np.array([record.get("split_position", idx) for idx, record in enumerate(records)])
    event_indices = np.array([record["event_idx"] for record in records])
    accuracies = np.array(
        [
            np.nan if record.get("accuracy") is None else float(record["accuracy"])
            for record in records
        ],
        dtype=float,
    )
    num_hits = np.array([int(record.get("num_hits", 0)) for record in records], dtype=float)
    incorrect_hits = np.array([int(record.get("incorrect_hits", 0)) for record in records], dtype=float)
    valid = np.isfinite(accuracies)
    if not bool(valid.any()):
        return

    mean_accuracy = float(np.nanmean(accuracies))
    median_accuracy = float(np.nanmedian(accuracies))
    min_accuracy = float(np.nanmin(accuracies))
    marker_sizes = np.clip(18.0 + np.sqrt(np.clip(num_hits, 0, None)) * 2.0, 18.0, 120.0)

    fig, (ax_scatter, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    scatter = ax_scatter.scatter(
        positions[valid],
        accuracies[valid],
        c=incorrect_hits[valid],
        s=marker_sizes[valid],
        cmap="Reds",
        alpha=0.82,
        edgecolors="#1f2933",
        linewidths=0.25,
    )
    ax_scatter.axhline(mean_accuracy, color="#1f77b4", linewidth=1.2, label=f"mean {mean_accuracy:.3f}")
    ax_scatter.axhline(
        median_accuracy,
        color="#2ca02c",
        linewidth=1.2,
        linestyle="--",
        label=f"median {median_accuracy:.3f}",
    )
    ax_scatter.set_ylim(-0.03, 1.03)
    ax_scatter.set_xlabel("event position in split")
    ax_scatter.set_ylabel("hit accuracy")
    ax_scatter.set_title(
        f"{title}\n"
        f"events={int(valid.sum())}, mean={mean_accuracy:.3f}, "
        f"median={median_accuracy:.3f}, min={min_accuracy:.3f}"
    )
    ax_scatter.grid(True, alpha=0.25)
    ax_scatter.legend(loc="lower right")
    colorbar = fig.colorbar(scatter, ax=ax_scatter, pad=0.01)
    colorbar.set_label("incorrect hits")

    worst_order = np.lexsort((-incorrect_hits[valid], accuracies[valid]))
    valid_positions = positions[valid]
    valid_accuracies = accuracies[valid]
    valid_event_indices = event_indices[valid]
    for order_idx in worst_order[: max(0, int(annotate_worst))]:
        ax_scatter.annotate(
            str(int(valid_event_indices[order_idx])),
            xy=(valid_positions[order_idx], valid_accuracies[order_idx]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=7,
            color="#7f1d1d",
        )

    bins = np.linspace(0.0, 1.0, 21)
    ax_hist.hist(accuracies[valid], bins=bins, color="#4c78a8", alpha=0.8)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("event hit accuracy")
    ax_hist.set_ylabel("events")
    ax_hist.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _finite_record_values(records, x_key, y_key="accuracy"):
    x_values = []
    y_values = []
    colors = []
    for record in records:
        x_value = record.get(x_key)
        y_value = record.get(y_key)
        if x_value is None or y_value is None:
            continue
        try:
            x_value = float(x_value)
            y_value = float(y_value)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            continue
        x_values.append(x_value)
        y_values.append(y_value)
        colors.append(float(record.get("incorrect_hits", 0)))
    return np.asarray(x_values), np.asarray(y_values), np.asarray(colors)


def plot_event_diagnostic_correlations(records, output_path, title):
    """Plot event accuracy against confidence and shower-overlap diagnostics."""
    if not records:
        return

    metric_specs = [
        ("loss", "event cross entropy"),
        ("num_hits", "ECal hits"),
        ("mean_confidence", "mean confidence"),
        ("mean_entropy", "mean normalized entropy"),
        ("energy_weighted_accuracy", "energy-weighted accuracy"),
        (
            "contributor_min_centroid_distance_xy",
            "any-contributor min centroid distance XY [mm]",
        ),
        (
            "contributor_min_normalized_shower_separation_xy",
            "any-contributor min normalized separation XY",
        ),
        (
            "dominant_min_normalized_shower_separation_xy",
            "dominant-only min normalized separation XY",
        ),
        (
            "early_contributor_min_normalized_shower_separation_xy",
            "first-three-layer any-contributor min normalized separation XY",
        ),
        (
            "early_dominant_min_normalized_shower_separation_xy",
            "first-three-layer dominant-only min normalized separation XY",
        ),
        ("ambiguous_hit_fraction_xy", "geometrically ambiguous hit fraction"),
        (
            "first_layer_contributor_min_centroid_distance_xy",
            "first-layer any-contributor min centroid distance XY [mm]",
        ),
        (
            "first_layer_dominant_min_centroid_distance_xy",
            "first-layer dominant-only min centroid distance XY [mm]",
        ),
        ("mean_hit_centroid_margin_xy", "mean hit centroid margin XY [mm]"),
    ]
    available = []
    for key, label in metric_specs:
        x_values, y_values, colors = _finite_record_values(records, key)
        if x_values.size >= 2:
            available.append((key, label, x_values, y_values, colors))

    if not available:
        return

    cols = 2
    rows = int(np.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.2 * rows), squeeze=False)
    scatter = None
    for ax, (key, label, x_values, y_values, colors) in zip(axes.flat, available):
        wrapped_label = "\n".join(textwrap.wrap(label, width=42))
        scatter = ax.scatter(
            x_values,
            y_values,
            c=colors,
            cmap="Reds",
            s=32,
            alpha=0.78,
            edgecolors="#1f2933",
            linewidths=0.25,
        )
        if x_values.size > 2 and np.nanstd(x_values) > 0 and np.nanstd(y_values) > 0:
            correlation = float(np.corrcoef(x_values, y_values)[0, 1])
            ax.text(
                0.03,
                0.05,
                f"r={correlation:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
        ax.set_xlabel(wrapped_label)
        ax.set_ylabel("event hit accuracy")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.set_title(wrapped_label)

    for ax in axes.flat[len(available):]:
        ax.axis("off")

    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(
        left=0.07,
        right=0.89,
        bottom=0.04,
        top=0.965,
        hspace=0.55,
        wspace=0.28,
    )
    if scatter is not None:
        colorbar_axis = fig.add_axes((0.915, 0.08, 0.018, 0.84))
        colorbar = fig.colorbar(scatter, cax=colorbar_axis)
        colorbar.set_label("incorrect hits")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _binned_record_profile(
    records,
    metric,
    target,
    num_bins,
    bootstrap_samples,
    seed,
):
    x_values, y_values, _colors = _finite_record_values(records, metric, y_key=target)
    edges = quantile_edges(x_values, num_bins=num_bins)
    if edges is None:
        return None

    points = []
    for bin_idx in range(edges.size - 1):
        low = float(edges[bin_idx])
        high = float(edges[bin_idx + 1])
        selected = (
            (x_values >= low) & (x_values <= high)
            if bin_idx == edges.size - 2
            else (x_values >= low) & (x_values < high)
        )
        if not bool(selected.any()):
            continue
        mean, ci_low, ci_high, _method = mean_confidence_interval(
            y_values[selected],
            bootstrap_samples=bootstrap_samples,
            seed=seed + bin_idx,
        )
        points.append(
            {
                "x": float(x_values[selected].mean()),
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "event_count": int(selected.sum()),
            }
        )
    return points or None


def plot_shower_separation_profiles(
    records,
    output_path,
    title,
    num_bins=8,
    bootstrap_samples=200,
    seed=7,
):
    """Plot accuracy against shower separation for each event multiplicity."""
    metric_specs = (
        (
            "contributor_min_normalized_shower_separation_xy",
            "any contributor, all layers",
            "#2563eb",
            "-",
        ),
        (
            "early_contributor_min_normalized_shower_separation_xy",
            "any contributor, first 3 layers",
            "#2563eb",
            "--",
        ),
        (
            "dominant_min_normalized_shower_separation_xy",
            "dominant only, all layers",
            "#d97706",
            "-",
        ),
        (
            "early_dominant_min_normalized_shower_separation_xy",
            "dominant only, first 3 layers",
            "#d97706",
            "--",
        ),
    )
    target_specs = (
        ("accuracy", "event hit accuracy"),
        ("energy_weighted_accuracy", "event energy-weighted accuracy"),
    )

    grouped = {}
    for record in records:
        multiplicity = record.get("electron_count")
        if multiplicity is None:
            multiplicity = record.get("num_truth_classes")
        try:
            multiplicity = int(multiplicity)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(multiplicity, []).append(record)
    multiplicities = sorted(grouped)
    if not multiplicities:
        return False

    fig, axes = plt.subplots(
        len(target_specs),
        len(multiplicities),
        figsize=(6.4 * len(multiplicities), 4.5 * len(target_specs)),
        squeeze=False,
        sharey="row",
    )
    plotted = False
    for col, multiplicity in enumerate(multiplicities):
        selected_records = grouped[multiplicity]
        for row, (target, target_label) in enumerate(target_specs):
            ax = axes[row, col]
            for metric_idx, (metric, metric_label, color, linestyle) in enumerate(metric_specs):
                points = _binned_record_profile(
                    selected_records,
                    metric=metric,
                    target=target,
                    num_bins=num_bins,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 1000 * row + 100 * col + 10 * metric_idx,
                )
                if not points:
                    continue
                x_values = np.asarray([point["x"] for point in points])
                means = np.asarray([point["mean"] for point in points])
                lows = np.clip(np.asarray([point["ci_low"] for point in points]), 0.0, 1.0)
                highs = np.clip(np.asarray([point["ci_high"] for point in points]), 0.0, 1.0)
                ax.plot(
                    x_values,
                    means,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    linewidth=1.7,
                    label=metric_label,
                )
                ax.fill_between(x_values, lows, highs, color=color, alpha=0.12)
                plotted = True

            ax.set_title(f"{multiplicity}e events (N={len(selected_records)})")
            ax.set_xlabel("minimum normalized shower separation (smaller = more overlap)")
            ax.set_ylabel(target_label)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
            if ax.lines:
                ax.legend(fontsize=8)

    if not plotted:
        plt.close(fig)
        return False
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_test_fraction_summaries(plot_samples, args, run_dir):
    if plot_samples is None or plot_samples["fraction_target"].numel() == 0:
        return

    target = plot_samples["fraction_target"].numpy()
    pred = plot_samples["fraction_pred"].numpy()
    valid_labels = tuple(args.valid_labels)

    fig, axes = plt.subplots(1, len(valid_labels), figsize=(4 * len(valid_labels), 4), sharex=True, sharey=True)
    axes = axes if isinstance(axes, (list, tuple)) else getattr(axes, "flat", [axes])
    for idx, (ax, origin) in enumerate(zip(axes, valid_labels)):
        ax.scatter(target[:, idx], pred[:, idx], s=5, alpha=0.35)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_title(f"origin {origin}")
        ax.set_xlabel("true fraction")
        if idx == 0:
            ax.set_ylabel("predicted fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Test-set origin energy fractions")
    fig.tight_layout()
    fig.savefig(run_dir / "test_fraction_scatter.png", dpi=200)
    plt.close(fig)

    target_max = target.max(axis=1)
    pred_max = pred.max(axis=1)
    bins = torch.linspace(0, 1, 31).numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(target_max, bins=bins, alpha=0.55, label="true max fraction")
    ax.hist(pred_max, bins=bins, alpha=0.55, label="predicted max fraction")
    ax.set_xlabel("max origin fraction per ECal hit")
    ax.set_ylabel("hits")
    ax.set_title("Test-set fraction purity")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "test_fraction_purity.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(plot_samples["per_hit_fraction_mae"].numpy(), bins=30, alpha=0.75)
    ax.set_xlabel("mean absolute fraction error per ECal hit")
    ax.set_ylabel("hits")
    ax.set_title("Test-set fraction MAE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "test_fraction_mae_hist.png", dpi=200)
    plt.close(fig)
