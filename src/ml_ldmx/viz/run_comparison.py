"""Report-oriented plots for paired hit-classifier run comparisons."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml_ldmx.eval.run_comparison import PROFILE_METRICS


MODEL_COLORS = ("#2563eb", "#d97706")


def plot_binned_profiles(profile_rows, labels, target, output_path, title):
    selected = [row for row in profile_rows if row.get("target") == target]
    metrics = [metric for metric in PROFILE_METRICS if any(row["metric"] == metric for row in selected)]
    if not metrics:
        return False

    cols = 2
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        for label_idx, label in enumerate(labels):
            points = sorted(
                [
                    row
                    for row in selected
                    if row["metric"] == metric and row["model"] == label
                ],
                key=lambda row: row["bin_index"],
            )
            if not points:
                continue
            x_values = np.asarray([row["x_mean"] for row in points], dtype=float)
            means = np.asarray([row["mean"] for row in points], dtype=float)
            lows = np.clip(
                np.asarray([row["ci_low"] for row in points], dtype=float),
                0.0,
                1.0,
            )
            highs = np.clip(
                np.asarray([row["ci_high"] for row in points], dtype=float),
                0.0,
                1.0,
            )
            color = MODEL_COLORS[label_idx % len(MODEL_COLORS)]
            ax.plot(x_values, means, marker="o", linewidth=1.8, color=color, label=label)
            ax.fill_between(x_values, lows, highs, color=color, alpha=0.16)
        ax.set_xlabel(PROFILE_METRICS[metric])
        ax.set_ylabel(selected[0]["target_label"])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend()

    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_paired_accuracy(rows, labels, slugs, output_path, title):
    first_label, second_label = labels
    first_field = f"{slugs[first_label]}_accuracy"
    second_field = f"{slugs[second_label]}_accuracy"
    delta_field = f"accuracy_delta_{slugs[first_label]}_minus_{slugs[second_label]}"
    selected = [
        row
        for row in rows
        if row.get(first_field) is not None and row.get(second_field) is not None
    ]
    if not selected:
        return False

    first = np.asarray([float(row[first_field]) for row in selected], dtype=float)
    second = np.asarray([float(row[second_field]) for row in selected], dtype=float)
    delta = np.asarray([float(row[delta_field]) for row in selected], dtype=float)
    overlap = np.asarray(
        [
            np.nan
            if row.get(
                "contributor_min_normalized_shower_separation_xy",
                row.get("min_normalized_shower_separation_xy"),
            )
            is None
            else float(
                row.get(
                    "contributor_min_normalized_shower_separation_xy",
                    row.get("min_normalized_shower_separation_xy"),
                )
            )
            for row in selected
        ],
        dtype=float,
    )

    fig, (ax_scatter, ax_delta) = plt.subplots(1, 2, figsize=(12, 5))
    if bool(np.isfinite(overlap).any()):
        fill_value = float(np.nanmedian(overlap))
        scatter = ax_scatter.scatter(
            second,
            first,
            c=np.nan_to_num(overlap, nan=fill_value),
            cmap="viridis",
            s=30,
            alpha=0.75,
            edgecolors="#1f2933",
            linewidths=0.25,
        )
        colorbar = fig.colorbar(scatter, ax=ax_scatter, pad=0.01)
        colorbar.set_label("any-contributor min normalized separation XY")
    else:
        ax_scatter.scatter(second, first, s=30, alpha=0.75, color=MODEL_COLORS[0])
    ax_scatter.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1)
    ax_scatter.set_xlim(-0.03, 1.03)
    ax_scatter.set_ylim(-0.03, 1.03)
    ax_scatter.set_xlabel(f"{second_label} event accuracy")
    ax_scatter.set_ylabel(f"{first_label} event accuracy")
    ax_scatter.grid(True, alpha=0.25)
    ax_scatter.set_title("paired event accuracy")

    ax_delta.hist(delta, bins=30, color="#4c78a8", alpha=0.82)
    ax_delta.axvline(0.0, color="#111827", linestyle="--", linewidth=1)
    ax_delta.axvline(
        float(delta.mean()),
        color="#b91c1c",
        linewidth=1.4,
        label=f"mean {delta.mean():.3f}",
    )
    ax_delta.set_xlabel(f"accuracy difference: {first_label} minus {second_label}")
    ax_delta.set_ylabel("events")
    ax_delta.grid(True, alpha=0.25)
    ax_delta.legend()
    ax_delta.set_title("paired accuracy difference")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_accuracy_by_multiplicity(profile_rows, labels, output_path, title):
    if not profile_rows:
        return False
    multiplicities = sorted({int(row["electron_count"]) for row in profile_rows})
    if not multiplicities:
        return False

    x_positions = np.arange(len(multiplicities), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    for label_idx, label in enumerate(labels):
        indexed = {
            int(row["electron_count"]): row
            for row in profile_rows
            if row["model"] == label
        }
        means = np.asarray(
            [indexed[count]["mean"] if count in indexed else np.nan for count in multiplicities],
            dtype=float,
        )
        lows = np.asarray(
            [indexed[count]["ci_low"] if count in indexed else np.nan for count in multiplicities],
            dtype=float,
        )
        highs = np.asarray(
            [indexed[count]["ci_high"] if count in indexed else np.nan for count in multiplicities],
            dtype=float,
        )
        offset = (label_idx - (len(labels) - 1) / 2.0) * width
        errors = np.vstack(
            [
                np.clip(means - lows, 0.0, None),
                np.clip(highs - means, 0.0, None),
            ]
        )
        ax.bar(
            x_positions + offset,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=MODEL_COLORS[label_idx % len(MODEL_COLORS)],
            alpha=0.82,
            label=label,
        )
    ax.set_xticks(x_positions, labels=[str(value) for value in multiplicities])
    ax.set_xlabel("electron count")
    ax.set_ylabel("mean event hit accuracy")
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True
