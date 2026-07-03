from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


_ORIGIN_COLORS = {
    0: "tab:gray",
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
}


def _to_numpy(array):
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _label_color(label):
    try:
        return _ORIGIN_COLORS.get(int(label), f"C{int(label) % 10}")
    except (TypeError, ValueError):
        return "C0"


def _prepare_ecal_hit_arrays(pos, true_labels, predicted_labels=None):
    pos = _to_numpy(pos)
    true_labels = _to_numpy(true_labels).reshape(-1)
    predicted_labels = (
        None
        if predicted_labels is None
        else _to_numpy(predicted_labels).reshape(-1)
    )

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected ECal hit positions with shape [N, 3], got {pos.shape}.")
    if true_labels.shape[0] != pos.shape[0]:
        raise ValueError(
            "Expected true labels to align with ECal hit positions: "
            f"{true_labels.shape[0]} labels for {pos.shape[0]} hits."
        )
    if predicted_labels is not None and predicted_labels.shape[0] != pos.shape[0]:
        raise ValueError(
            "Expected predicted labels to align with ECal hit positions: "
            f"{predicted_labels.shape[0]} labels for {pos.shape[0]} hits."
        )
    return pos, true_labels, predicted_labels


def _prepare_optional_hit_labels(name, labels, num_hits):
    if labels is None:
        return None
    labels = _to_numpy(labels).reshape(-1)
    if labels.shape[0] != num_hits:
        raise ValueError(
            f"Expected {name} to align with ECal hit positions: "
            f"{labels.shape[0]} labels for {num_hits} hits."
        )
    return labels


def plot_ecal_hit_classes_3d(pos, physical_labels, output_path, title, labels=None):
    """Save a 3D ECal hit scatter plot colored by origin/class labels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos, physical_labels, _ = _prepare_ecal_hit_arrays(pos, physical_labels)
    labels = [1, 2, 3] if labels is None else list(labels)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label in labels:
        mask = physical_labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            pos[mask, 0],
            pos[mask, 1],
            pos[mask, 2],
            s=10,
            alpha=0.85,
            color=_label_color(label),
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=_label_color(label),
            label=str(label),
        )
        for label in labels
    ]
    ax.legend(handles=handles, title="origin_id")
    ax.set_title(title)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(200, 700)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ecal_hit_prediction_errors_3d(
    pos,
    true_labels,
    predicted_labels,
    output_path,
    title,
    labels=None,
    color_labels=None,
    legend_title="true origin/class",
):
    """Save one 3D ECal plot: true-label colors with red X markers on wrong predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos, true_labels, predicted_labels = _prepare_ecal_hit_arrays(
        pos,
        true_labels,
        predicted_labels,
    )
    color_labels = _prepare_optional_hit_labels(
        "color_labels",
        color_labels,
        pos.shape[0],
    )
    color_labels = true_labels if color_labels is None else color_labels
    if labels is None:
        labels = sorted(np.unique(color_labels).tolist())
    else:
        labels = list(labels)

    correct_mask = true_labels == predicted_labels
    wrong_mask = ~correct_mask
    wrong_count = int(wrong_mask.sum())
    total_hits = int(true_labels.shape[0])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label in labels:
        label_mask = color_labels == label
        correct_label_mask = label_mask & correct_mask
        wrong_label_mask = label_mask & wrong_mask
        color = _label_color(label)

        if np.any(correct_label_mask):
            ax.scatter(
                pos[correct_label_mask, 0],
                pos[correct_label_mask, 1],
                pos[correct_label_mask, 2],
                s=10,
                alpha=0.82,
                color=color,
            )
        if np.any(wrong_label_mask):
            ax.scatter(
                pos[wrong_label_mask, 0],
                pos[wrong_label_mask, 1],
                pos[wrong_label_mask, 2],
                s=18,
                alpha=0.95,
                color=color,
                edgecolors="black",
                linewidths=0.35,
            )
            ax.scatter(
                pos[wrong_label_mask, 0],
                pos[wrong_label_mask, 1],
                pos[wrong_label_mask, 2],
                s=46,
                alpha=1.0,
                color="red",
                marker="x",
                linewidths=1.8,
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=_label_color(label),
            label=str(label),
        )
        for label in labels
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="",
            color="red",
            markeredgewidth=1.8,
            label="incorrect prediction",
        )
    )

    accuracy = 1.0 - wrong_count / max(1, total_hits)
    ax.legend(handles=handles, title=legend_title)
    ax.set_title(
        f"{title}\nincorrect hits: {wrong_count}/{total_hits} ({accuracy:.1%} correct)"
    )
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(200, 700)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ecal_truth_prediction_pair(
    pos,
    true_labels,
    predicted_labels,
    truth_path,
    predicted_path,
    truth_title,
    predicted_title,
    labels=None,
):
    """Save matching truth and predicted ECal class plots for one event."""
    plot_ecal_hit_classes_3d(
        pos,
        true_labels,
        truth_path,
        truth_title,
        labels=labels,
    )
    plot_ecal_hit_classes_3d(
        pos,
        predicted_labels,
        predicted_path,
        predicted_title,
        labels=labels,
    )
