import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _confusion_from_labels(y_true, y_pred, labels):
    label_to_index = {int(label): idx for idx, label in enumerate(labels)}
    confusion = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    for true_value, pred_value in zip(y_true, y_pred):
        true_idx = label_to_index.get(int(true_value))
        pred_idx = label_to_index.get(int(pred_value))
        if true_idx is None or pred_idx is None:
            continue
        confusion[true_idx, pred_idx] += 1
    return confusion


def plot_event_count_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title="Event electron count confusion matrix",
    output_path=None,
    normalize=False,
):
    if labels is None:
        labels = sorted({int(value) for value in list(y_true) + list(y_pred)})
    labels = [int(label) for label in labels]
    confusion = _confusion_from_labels(y_true, y_pred, labels).to(dtype=torch.float64)

    if normalize:
        row_sums = confusion.sum(dim=1, keepdim=True).clamp_min(1.0)
        image_values = confusion / row_sums
        colorbar_label = "row-normalized fraction"
        vmin, vmax = 0.0, 1.0
    else:
        image_values = confusion
        colorbar_label = "events"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(image_values.numpy(), cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("predicted count")
    ax.set_ylabel("true count")
    ax.set_xticks(range(len(labels)), labels=[str(label) for label in labels])
    ax.set_yticks(range(len(labels)), labels=[str(label) for label in labels])

    max_value = float(image_values.max().item()) if image_values.numel() else 0.0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            count = int(confusion[row, col].item())
            value = float(image_values[row, col].item())
            text = f"{count}\n{value:.2f}" if normalize else str(count)
            text_color = "white" if max_value > 0.0 and value > 0.5 * max_value else "black"
            ax.text(col, row, text, ha="center", va="center", color=text_color)

    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig, ax
