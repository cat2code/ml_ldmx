"""Paired TPad-ablation plots for the contributor-set slot model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_COLOR = "#2563eb"
ABLATED_COLOR = "#d97706"


def _normalized_confusion(confusion):
    values = np.asarray(confusion, dtype=float)
    row_sum = values.sum(axis=1, keepdims=True)
    return values / np.maximum(row_sum, 1.0)


def plot_confusion_ablation(
    reference_confusion,
    ablated_confusion,
    labels,
    output_path,
    title,
):
    """Plot matched row-normalized confusion matrices with raw counts."""
    reference = np.asarray(reference_confusion, dtype=int)
    ablated = np.asarray(ablated_confusion, dtype=int)
    if reference.shape != ablated.shape or reference.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrices and labels must have matching square shapes.")
    normalized = [_normalized_confusion(reference), _normalized_confusion(ablated)]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = max(12, 1.4 * len(labels) * 2)
    fig, axes = plt.subplots(1, 2, figsize=(width, max(5, 1.15 * len(labels))))
    image = None
    for ax, counts, fractions, subtitle in zip(
        axes,
        (reference, ablated),
        normalized,
        ("with TPad", "TPad removed"),
    ):
        image = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues", origin="upper")
        ax.set_title(subtitle)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        annotation_size = 7 if len(labels) > 5 else 9
        for row in range(len(labels)):
            for col in range(len(labels)):
                fraction = float(fractions[row, col])
                ax.text(
                    col,
                    row,
                    f"{int(counts[row, col])}\n{fraction:.2f}",
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    color="white" if fraction > 0.5 else "black",
                )
    fig.colorbar(image, ax=axes, shrink=0.82, label="row-normalized fraction")
    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.17, top=0.87, wspace=0.28)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig


def plot_task_metric_ablation(reference, ablated, output_path):
    """Compare all reconstructed tasks using metrics with compatible scales."""
    accuracy_metrics = [
        ("count_accuracy", "count"),
        ("count_accuracy_2e", "count, true 2e"),
        ("count_accuracy_3e", "count, true 3e"),
        ("slot_exact_accuracy", "slot exact"),
        ("support_accuracy", "contributor set"),
        ("mixed_f1", "mixed-hit F1"),
        ("origin_accuracy", "dominant origin"),
    ]
    error_metrics = [
        ("fraction_mae", "fraction MAE"),
        ("raw_fraction_mae", "raw fraction MAE"),
        ("mixed_brier", "mixed-probability Brier"),
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_accuracy, ax_error) = plt.subplots(2, 1, figsize=(12, 9))

    def grouped_bars(ax, specifications, ylabel, limits, *, lower_is_better=False):
        available = [item for item in specifications if item[0] in reference and item[0] in ablated]
        positions = np.arange(len(available))
        width = 0.36
        ref_values = [float(reference[key]) for key, _label in available]
        abl_values = [float(ablated[key]) for key, _label in available]
        ax.bar(positions - width / 2, ref_values, width, color=REFERENCE_COLOR, label="with TPad")
        ax.bar(positions + width / 2, abl_values, width, color=ABLATED_COLOR, label="TPad removed")
        ax.set_xticks(positions, labels=[label for _key, label in available], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*limits)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        for position, ref_value, abl_value in zip(positions, ref_values, abl_values):
            tpad_gain = (
                abl_value - ref_value
                if lower_is_better
                else ref_value - abl_value
            )
            ax.text(
                position,
                max(ref_value, abl_value) + 0.025 * (limits[1] - limits[0]),
                f"TPad gain={tpad_gain:+.3f}",
                ha="center",
                fontsize=8,
            )

    grouped_bars(ax_accuracy, accuracy_metrics, "accuracy / F1", (0.0, 1.08))
    max_error = max(
        [float(reference[key]) for key, _ in error_metrics if key in reference]
        + [float(ablated[key]) for key, _ in error_metrics if key in ablated]
        + [0.1]
    )
    grouped_bars(
        ax_error,
        error_metrics,
        "error (lower is better)",
        (0.0, 1.18 * max_error),
        lower_is_better=True,
    )
    ax_accuracy.set_title("Classification and event-reconstruction performance")
    ax_error.set_title("Fraction and probability errors")
    fig.suptitle("Paired TPad ablation across contributor-slot tasks")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig


def _prediction_pairs(reference_predictions, ablated_predictions):
    reference = {int(row["event_index"]): row for row in reference_predictions}
    ablated = {int(row["event_index"]): row for row in ablated_predictions}
    if set(reference) != set(ablated):
        raise ValueError("Reference and ablated predictions must contain identical events.")
    return [(reference[index], ablated[index]) for index in sorted(reference)]


def plot_count_ablation(reference_predictions, ablated_predictions, output_path):
    """Visualize how TPad removal changes the derived 2e/3e decision."""
    pairs = _prediction_pairs(reference_predictions, ablated_predictions)
    if not pairs:
        return None
    true_count = np.asarray([int(ref["true_count"]) for ref, _abl in pairs])
    ref_count = np.asarray([int(ref["predicted_count"]) for ref, _abl in pairs])
    abl_count = np.asarray([int(abl["predicted_count"]) for _ref, abl in pairs])
    ref_slot3 = np.asarray([float(ref["slot_probability"][2]) for ref, _abl in pairs])
    abl_slot3 = np.asarray([float(abl["slot_probability"][2]) for _ref, abl in pairs])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    labels = ("all", "true 2e", "true 3e")
    selections = (
        np.ones_like(true_count, dtype=bool),
        true_count == 2,
        true_count == 3,
    )
    positions = np.arange(len(labels))
    width = 0.36
    ref_accuracy = [float((ref_count[selected] == true_count[selected]).mean()) for selected in selections]
    abl_accuracy = [float((abl_count[selected] == true_count[selected]).mean()) for selected in selections]
    ax.bar(positions - width / 2, ref_accuracy, width, color=REFERENCE_COLOR, label="with TPad")
    ax.bar(positions + width / 2, abl_accuracy, width, color=ABLATED_COLOR, label="TPad removed")
    ax.set_xticks(positions, labels=labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("electron-count accuracy")
    ax.set_title("Derived electron count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for count, color in ((2, "#0f766e"), (3, "#b91c1c")):
        selected = true_count == count
        ax.scatter(
            ref_slot3[selected],
            abl_slot3[selected],
            s=7,
            alpha=0.12,
            edgecolors="none",
            color=color,
            label=f"true {count}e",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4b5563")
    ax.axvline(0.5, color="#9ca3af", linewidth=0.8)
    ax.axhline(0.5, color="#9ca3af", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("slot-3 probability with TPad")
    ax.set_ylabel("slot-3 probability with TPad removed")
    ax.set_title("Paired slot-3 confidence")
    ax.grid(alpha=0.2)
    ax.legend()

    ax = axes[1, 0]
    probability_gain = ref_slot3 - abl_slot3
    bins = np.linspace(-1, 1, 51)
    for count, color in ((2, "#0f766e"), (3, "#b91c1c")):
        selected = true_count == count
        ax.hist(
            probability_gain[selected],
            bins=bins,
            alpha=0.6,
            color=color,
            label=f"true {count}e",
        )
    ax.axvline(0, color="#4b5563", linewidth=1)
    ax.set_xlabel("slot-3 probability: with TPad − removed")
    ax.set_ylabel("events")
    ax.set_title("Change in electron-3 evidence")
    ax.grid(alpha=0.2)
    ax.legend()

    ax = axes[1, 1]
    ref_correct = ref_count == true_count
    abl_correct = abl_count == true_count
    categories = (
        ("both correct", ref_correct & abl_correct),
        ("only with TPad", ref_correct & ~abl_correct),
        ("only without TPad", ~ref_correct & abl_correct),
        ("both wrong", ~ref_correct & ~abl_correct),
    )
    values = [int(selected.sum()) for _label, selected in categories]
    bars = ax.bar(
        np.arange(len(categories)),
        values,
        color=("#64748b", REFERENCE_COLOR, ABLATED_COLOR, "#991b1b"),
    )
    ax.set_xticks(np.arange(len(categories)), labels=[label for label, _ in categories], rotation=18, ha="right")
    ax.set_ylabel("events")
    ax.set_title("Per-event count outcome changes")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars)

    fig.suptitle("Electron-count reliance on TPad information")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig


def plot_fraction_mae_ablation(reference_plot_data, ablated_plot_data, output_path):
    """Compare fraction MAE by output component on the same sampled hits."""
    target = np.asarray(reference_plot_data["fraction_target"], dtype=float)
    reference = np.asarray(reference_plot_data["fraction_pred"], dtype=float)
    ablated_target = np.asarray(ablated_plot_data["fraction_target"], dtype=float)
    ablated = np.asarray(ablated_plot_data["fraction_pred"], dtype=float)
    if target.shape != reference.shape or target.shape != ablated.shape:
        raise ValueError("Reference and ablated fraction samples must align.")
    if not np.allclose(target, ablated_target):
        raise ValueError("TPad ablation changed the sampled fraction targets.")
    labels = ["noise"] + [f"electron {index}" for index in range(1, target.shape[1])]
    ref_mae = np.abs(reference - target).mean(axis=0)
    abl_mae = np.abs(ablated - target).mean(axis=0)
    positions = np.arange(len(labels))
    width = 0.36
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, ref_mae, width, color=REFERENCE_COLOR, label="with TPad")
    ax.bar(positions + width / 2, abl_mae, width, color=ABLATED_COLOR, label="TPad removed")
    ax.set_xticks(positions, labels=labels)
    ax.set_ylabel("mean absolute fraction error")
    ax.set_title("Energy-fraction reconstruction under TPad ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return fig
