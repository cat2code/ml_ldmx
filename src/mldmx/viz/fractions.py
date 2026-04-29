from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_fraction_scatter(fraction_target, fraction_pred, output_path, valid_labels=(1, 2, 3)):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = np.asarray(fraction_target)
    pred = np.asarray(fraction_pred)
    fig, axes = plt.subplots(1, len(valid_labels), figsize=(4 * len(valid_labels), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for idx, (ax, origin) in enumerate(zip(axes, valid_labels)):
        ax.scatter(target[:, idx], pred[:, idx], s=8, alpha=0.55)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_title(f"origin {origin}")
        ax.set_xlabel("true fraction")
        if idx == 0:
            ax.set_ylabel("predicted fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Event 9 ECal origin energy fractions")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_fraction_purity(fraction_target, fraction_pred, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_max = np.asarray(fraction_target).max(axis=1)
    pred_max = np.asarray(fraction_pred).max(axis=1)
    bins = np.linspace(0, 1, 31)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(target_max, bins=bins, alpha=0.55, label="true max fraction")
    ax.hist(pred_max, bins=bins, alpha=0.55, label="predicted max fraction")
    ax.set_xlabel("max origin fraction per ECal hit")
    ax.set_ylabel("hits")
    ax.set_title("Event 9 fraction purity")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_fraction_mae_hist(per_hit_mae, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.asarray(per_hit_mae), bins=30, alpha=0.75)
    ax.set_xlabel("mean absolute fraction error per ECal hit")
    ax.set_ylabel("hits")
    ax.set_title("Event 9 fraction MAE")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_fraction_error_3d(pos, per_hit_mae, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(pos)
    per_hit_mae = np.asarray(per_hit_mae)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        c=per_hit_mae,
        s=10,
        alpha=0.85,
        cmap="viridis",
    )
    ax.set_title("Event 9 ECal hits, origin-fraction MAE")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(200, 700)
    fig.colorbar(scatter, ax=ax, shrink=0.65, label="fraction MAE")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
