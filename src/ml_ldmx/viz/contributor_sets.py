"""Plots for contributor-set and learned mixed-hit reconstruction."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def contributor_set_labels(max_electrons: int) -> list[str]:
    """Return compact human-readable labels in bit-mask class order."""
    labels = []
    for value in range(1 << max_electrons):
        contributors = [str(index + 1) for index in range(max_electrons) if value & (1 << index)]
        labels.append("noise" if not contributors else "+".join(contributors))
    return labels


def plot_mixed_probability_diagnostics(
    mixed_target,
    mixed_probability,
    output_path,
    *,
    title="Learned mixed-hit probability",
    num_bins: int = 10,
):
    """Plot score separation and empirical calibration without a hand threshold."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = np.asarray(mixed_target, dtype=bool).reshape(-1)
    probability = np.asarray(mixed_probability, dtype=float).reshape(-1)
    finite = np.isfinite(probability)
    target = target[finite]
    probability = np.clip(probability[finite], 0.0, 1.0)
    if probability.size == 0:
        return None

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.clip(np.digitize(probability, edges[1:-1]), 0, num_bins - 1)
    observed = np.full(num_bins, np.nan)
    counts = np.zeros(num_bins, dtype=int)
    for index in range(num_bins):
        selected = bin_index == index
        counts[index] = int(selected.sum())
        if counts[index]:
            observed[index] = float(target[selected].mean())

    fig, (ax_score, ax_calibration) = plt.subplots(1, 2, figsize=(11, 4.5))
    for selected, label in (
        (~target, "true pure/noise"),
        (target, "true mixed"),
    ):
        if selected.any():
            ax_score.hist(
                probability[selected],
                bins=edges,
                alpha=0.7,
                density=True,
                label=label,
            )
    ax_score.set_xlabel("predicted probability of multiple contributors")
    ax_score.set_ylabel("density")
    ax_score.grid(alpha=0.25)
    if ax_score.has_data():
        ax_score.legend()

    populated = counts > 0
    ax_calibration.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax_calibration.plot(
        centers[populated],
        observed[populated],
        marker="o",
        color="#4c78a8",
    )
    for x_value, y_value, count in zip(
        centers[populated], observed[populated], counts[populated]
    ):
        ax_calibration.annotate(str(count), (x_value, y_value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax_calibration.set_xlim(0, 1)
    ax_calibration.set_ylim(0, 1)
    ax_calibration.set_xlabel("mean predicted mixed probability")
    ax_calibration.set_ylabel("observed mixed fraction")
    ax_calibration.set_title("Calibration (labels show hits/bin)")
    ax_calibration.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig


def plot_contributor_set_history(history, output_path):
    """Plot validation-relevant tasks separately from the aggregate objective."""
    if not history:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    specifications = [
        ("loss", "aggregate loss", None),
        ("support_accuracy", "contributor-set accuracy", (0, 1)),
        ("mixed_f1", "mixed-hit F1", (0, 1)),
        ("count_accuracy", "derived event-count accuracy", (0, 1)),
    ]
    for ax, (metric, label, limits) in zip(axes.flat, specifications):
        for split, color in (("train", "#4c78a8"), ("val", "#f58518")):
            key = f"{split}_{metric}"
            values = [row.get(key, np.nan) for row in history]
            ax.plot(epochs, values, marker="o", label=split, color=color)
        ax.set_ylabel(label)
        if limits is not None:
            ax.set_ylim(*limits)
        ax.grid(alpha=0.25)
        ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    fig.suptitle("Contributor-set slot model training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig


def plot_fraction_reconstruction(
    fraction_target,
    fraction_prediction,
    output_path,
    *,
    electron_labels=(1, 2, 3),
):
    """Compare reconstructed and true electron fractions on sampled test hits."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = np.asarray(fraction_target, dtype=float)
    prediction = np.asarray(fraction_prediction, dtype=float)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Fraction target and prediction must have matching [hits, slots] shapes.")
    if target.shape[1] != len(electron_labels):
        raise ValueError("electron_labels must align with the fraction columns.")

    fig, axes = plt.subplots(
        1,
        len(electron_labels),
        figsize=(4 * len(electron_labels), 4),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for index, (ax, label) in enumerate(zip(axes, electron_labels)):
        ax.hexbin(
            target[:, index],
            prediction[:, index],
            gridsize=35,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_title(f"electron {label}")
        ax.set_xlabel("true fraction")
        if index == 0:
            ax.set_ylabel("reconstructed fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    fig.suptitle("Test-hit electron energy-fraction reconstruction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig
