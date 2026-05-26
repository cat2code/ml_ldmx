import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    image = ax.imshow(normalized.numpy(), vmin=0, vmax=1, cmap="Blues")
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
