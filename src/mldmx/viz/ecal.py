from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_ecal_hit_classes_3d(pos, physical_labels, output_path, title):
    """Save a 3D ECal hit scatter plot colored by physical labels 1, 2, 3."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(pos)
    physical_labels = np.asarray(physical_labels)
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label in [1, 2, 3]:
        mask = physical_labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            pos[mask, 0],
            pos[mask, 1],
            pos[mask, 2],
            s=10,
            alpha=0.85,
            color=colors[label],
        )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=str(label))
        for label in [1, 2, 3]
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
