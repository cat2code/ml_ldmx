import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


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
        ("min_origin_centroid_distance_xy", "min origin centroid distance XY [mm]"),
        (
            "min_normalized_shower_separation_xy",
            "min normalized shower separation XY",
        ),
        (
            "early_min_normalized_shower_separation_xy",
            "early-layer min normalized shower separation XY",
        ),
        ("ambiguous_hit_fraction_xy", "geometrically ambiguous hit fraction"),
        (
            "first_layer_min_origin_centroid_distance_xy",
            "first-layer min origin centroid distance XY [mm]",
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
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows), squeeze=False)
    scatter = None
    for ax, (key, label, x_values, y_values, colors) in zip(axes.flat, available):
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
        ax.set_xlabel(label)
        ax.set_ylabel("event hit accuracy")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.set_title(label)

    for ax in axes.flat[len(available):]:
        ax.axis("off")

    fig.suptitle(title)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes[:, :], shrink=0.88, pad=0.015)
        colorbar.set_label("incorrect hits")
    fig.subplots_adjust(top=0.91, hspace=0.38, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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
