"""
Run from the ml_ldmx directory:

    cd ml_ldmx
    python3 -m pip install -e .
    python3 scripts/simple_3_class_classification_ecal_transformer.py

Example with explicit options:

    python3 scripts/simple_3_class_classification_ecal_transformer.py \
        --root-file data/28apr_00/events.root \
        --epochs 30 \
        --d-model 64 \
        --lr 1e-3
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from ml_ldmx.datasets.tensorize import tensorize_ecal_node_classification
from ml_ldmx.io.root_reader import read_ecal_rechits_with_truth
from ml_ldmx.models import ECalHitTransformer
from ml_ldmx.viz.ecal import plot_ecal_hit_classes_3d


VALID_LABELS = (1, 2, 3)


def event_to_tensors(event, event_idx, filter_noise=True):
    tensors = tensorize_ecal_node_classification(
        event,
        valid_labels=VALID_LABELS,
        filter_noise=filter_noise,
    )
    tensors["event_idx"] = event_idx
    return tensors


def count_classes(events):
    counter = Counter()
    for event in events:
        counter.update(event["physical_y"].tolist())
    return dict(sorted(counter.items()))


def choose_device(requested_device):
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-file",
        default=project_root / "data/28apr_00/events.root",
        type=Path,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help="Use CPU by default for reproducible smoke tests; pass auto to use CUDA/MPS if available.",
    )
    parser.add_argument("--keep-noise", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root_file = args.root_file.resolve()
    model_path = project_root / "models/simple_3_class_ecal_transformer.pt"
    pred_plot_path = project_root / "figures/simple_3_class_event9_transformer_predicted.png"
    truth_plot_path = project_root / "figures/simple_3_class_event9_transformer_truth.png"

    print(f"Reading ROOT file: {root_file}")
    events = read_ecal_rechits_with_truth(root_file, max_events=10)
    if len(events) != 10:
        raise ValueError(f"Expected exactly 10 events, found {len(events)}.")

    filter_noise = not args.keep_noise
    print(
        "Noise handling: "
        + ("filtering out noise hits before training/evaluation" if filter_noise else "keeping noise hits")
    )

    tensor_events = []
    for event_idx, event in enumerate(events):
        n_noise = sum(bool(v) for v in event["noise_flag"])
        print(f"event {event_idx}: raw ECal hits={len(event['x'])}, noise_hits={n_noise}")
        tensor_event = event_to_tensors(event, event_idx=event_idx, filter_noise=filter_noise)
        tensor_events.append(tensor_event)
        print(
            f"event {event_idx}: selected_hits={tensor_event['x'].shape[0]}, "
            f"labels={sorted(set(tensor_event['physical_y'].tolist()))}"
        )

    train_events = tensor_events[:9]
    test_event = tensor_events[9]

    unique_labels = sorted({label for event in tensor_events for label in event["physical_y"].tolist()})
    if unique_labels != list(VALID_LABELS):
        raise ValueError(
            f"Expected physical labels {VALID_LABELS}, but saw {unique_labels}. "
            "Check that origin_id_contribs contains the intended 3-class labels."
        )

    print(f"training events: 0-8 ({len(train_events)} events)")
    print("evaluation event: 9")
    print("training class counts:", count_classes(train_events))
    print("unique labels seen:", unique_labels)

    device = choose_device(args.device)
    print(f"device: {device}")
    model = ECalHitTransformer(
        in_dim=train_events[0]["x"].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        out_dim=3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_hits = 0
        for event in train_events:
            x = event["x"].to(device)
            y = event["y"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_hits += x.shape[0]

        print(f"epoch={epoch:03d} train_loss={total_loss / total_hits:.4f}")

    model.eval()
    with torch.no_grad():
        x = test_event["x"].to(device)
        logits = model(x)
        pred_class = logits.argmax(dim=1).cpu()
        true_class = test_event["y"].cpu()
        accuracy = (pred_class == true_class).float().mean().item()

    class_to_label = {0: 1, 1: 2, 2: 3}
    pred_physical = torch.tensor([class_to_label[int(v)] for v in pred_class], dtype=torch.long)
    true_physical = torch.tensor([class_to_label[int(v)] for v in true_class], dtype=torch.long)
    print(f"event 9 test accuracy: {accuracy:.3f}")
    print("event 9 true class counts:", dict(sorted(Counter(true_physical.tolist()).items())))
    print("event 9 predicted class counts:", dict(sorted(Counter(pred_physical.tolist()).items())))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "valid_labels": VALID_LABELS,
            "class_to_label": class_to_label,
            "model_kwargs": {
                "in_dim": train_events[0]["x"].shape[1],
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "out_dim": 3,
            },
        },
        model_path,
    )
    print(f"saved model: {model_path}")

    pos = test_event["pos"].cpu()
    plot_ecal_hit_classes_3d(
        pos,
        pred_physical,
        pred_plot_path,
        "Event 9 ECal hits, transformer predicted dominant origin_id",
    )
    plot_ecal_hit_classes_3d(
        pos,
        true_physical,
        truth_plot_path,
        "Event 9 ECal hits, true dominant origin_id",
    )
    print(f"saved prediction plot: {pred_plot_path}")
    print(f"saved truth plot: {truth_plot_path}")


if __name__ == "__main__":
    main()
