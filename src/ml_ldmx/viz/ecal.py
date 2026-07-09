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


def _prepare_optional_hit_values(name, values, num_hits):
    if values is None:
        return None
    values = _to_numpy(values).reshape(-1)
    if values.shape[0] != num_hits:
        raise ValueError(
            f"Expected {name} to align with ECal hit positions: "
            f"{values.shape[0]} values for {num_hits} hits."
        )
    return values.astype(float)


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


def make_ecal_hit_prediction_figure_3d(
    pos,
    true_labels,
    predicted_labels=None,
    energy=None,
    confidence=None,
    entropy=None,
    title="Interactive ECal hit display",
):
    """Return an interactive Plotly 3D ECal hit display."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install.
        raise ImportError(
            "Plotly is required for interactive ECal displays. Install the project "
            "requirements or run `pip install plotly`."
        ) from exc

    pos, true_labels, predicted_labels = _prepare_ecal_hit_arrays(
        pos,
        true_labels,
        predicted_labels,
    )
    num_hits = pos.shape[0]
    energy = _prepare_optional_hit_values("energy", energy, num_hits)
    confidence = _prepare_optional_hit_values("confidence", confidence, num_hits)
    entropy = _prepare_optional_hit_values("entropy", entropy, num_hits)

    if predicted_labels is None:
        predicted_for_hover = np.full((num_hits,), np.nan)
        correctness = None
    else:
        predicted_for_hover = predicted_labels.astype(float)
        correctness = (predicted_labels == true_labels).astype(float)

    energy_for_hover = np.full((num_hits,), np.nan) if energy is None else energy
    confidence_for_hover = np.full((num_hits,), np.nan) if confidence is None else confidence
    entropy_for_hover = np.full((num_hits,), np.nan) if entropy is None else entropy
    customdata = np.column_stack(
        [
            true_labels.astype(float),
            predicted_for_hover,
            energy_for_hover,
            confidence_for_hover,
            entropy_for_hover,
        ]
    )
    hovertemplate = (
        "x=%{x:.1f} mm<br>"
        "y=%{y:.1f} mm<br>"
        "z=%{z:.1f} mm<br>"
        "truth=%{customdata[0]:.0f}<br>"
        "prediction=%{customdata[1]:.0f}<br>"
        "energy=%{customdata[2]:.4g}<br>"
        "confidence=%{customdata[3]:.3f}<br>"
        "entropy=%{customdata[4]:.3f}"
        "<extra></extra>"
    )

    traces = []
    trace_modes = []

    def add_trace(name, color, colorbar_title, colorscale="Turbo", visible=False, symbol="circle"):
        traces.append(
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="markers",
                name=name,
                visible=visible,
                customdata=customdata,
                hovertemplate=hovertemplate,
                marker={
                    "size": 3.5,
                    "color": color,
                    "colorscale": colorscale,
                    "colorbar": {"title": colorbar_title},
                    "opacity": 0.82,
                    "symbol": symbol,
                },
            )
        )
        trace_modes.append((name, [len(traces) - 1]))

    add_trace("truth origin", true_labels.astype(float), "truth", visible=True)
    if predicted_labels is not None:
        add_trace("predicted origin", predicted_labels.astype(float), "prediction")
        add_trace(
            "correctness",
            correctness,
            "correct",
            colorscale=[[0.0, "#d62728"], [0.499, "#d62728"], [0.5, "#2ca02c"], [1.0, "#2ca02c"]],
        )
        wrong_mask = correctness == 0.0
        if bool(np.any(wrong_mask)):
            wrong_trace_idx = len(traces)
            traces.append(
                go.Scatter3d(
                    x=pos[wrong_mask, 0],
                    y=pos[wrong_mask, 1],
                    z=pos[wrong_mask, 2],
                    mode="markers",
                    name="incorrect hits",
                    visible=False,
                    customdata=customdata[wrong_mask],
                    hovertemplate=hovertemplate,
                    marker={
                        "size": 6.5,
                        "color": "#d62728",
                        "opacity": 0.95,
                        "symbol": "diamond",
                    },
                )
            )
            trace_modes.append(("truth + incorrect", [0, wrong_trace_idx]))
            trace_modes.append(("incorrect only", [wrong_trace_idx]))
    if confidence is not None:
        add_trace("confidence", confidence, "confidence", colorscale="Viridis")
    if entropy is not None:
        add_trace("entropy", entropy, "entropy", colorscale="Magma")
    if energy is not None:
        add_trace("energy", energy, "energy", colorscale="Plasma")

    buttons = []
    for mode_name, visible_indices in trace_modes:
        visible = [False] * len(traces)
        for idx in visible_indices:
            visible[idx] = True
        buttons.append(
            {
                "label": mode_name,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"{title} - {mode_name}"},
                ],
            }
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "X [mm]",
            "yaxis_title": "Y [mm]",
            "zaxis_title": "Z [mm]",
            "xaxis": {"range": [-300, 300]},
            "yaxis": {"range": [-300, 300]},
            "zaxis": {"range": [200, 700]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.85},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 58},
        updatemenus=[
            {
                "type": "dropdown",
                "x": 0.0,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": buttons,
            }
        ],
    )
    return fig


def plot_ecal_hit_prediction_errors_3d_interactive(
    pos,
    true_labels,
    predicted_labels,
    output_path=None,
    title="Interactive ECal hit prediction errors",
    energy=None,
    confidence=None,
    entropy=None,
):
    """Create an interactive 3D prediction display and optionally save it as HTML."""
    fig = make_ecal_hit_prediction_figure_3d(
        pos=pos,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        energy=energy,
        confidence=confidence,
        entropy=entropy,
        title=title,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path, include_plotlyjs="cdn")
    return fig
