"""
Run from the ml_ldmx directory:

    cd ml_ldmx
    python3 -m pip install -e .
    python3 scripts/simple_3_class_classification_tiny_gnn.py

Example with explicit options:

    python3 -m pip install -e .;
    python3 scripts/simple_3_class_classification_tiny_gnn.py \
        --root-file data/28apr_00/events.root \
        --epochs 30 \
        --hidden-dim 32 \
        --lr 1e-3
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv

from ml_ldmx.datasets.graph_builder import build_knn_graph
from ml_ldmx.datasets.tensorize import tensorize_ecal_node_classification
from ml_ldmx.io.root_reader import read_ecal_rechits_with_truth
from ml_ldmx.viz.ecal import plot_ecal_hit_classes_3d


VALID_LABELS = (1, 2, 3)


class TinyNodeGNN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=32, out_dim=3):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.head(x)


def event_to_graph(event, event_idx, filter_noise=True):
    tensors = tensorize_ecal_node_classification(
        event,
        valid_labels=VALID_LABELS,
        filter_noise=filter_noise,
    )
    edge_index = build_knn_graph(tensors["pos"], k=8)
    return Data(
        x=tensors["x"],
        pos=tensors["pos"],
        edge_index=edge_index,
        y=tensors["y"],
        physical_y=tensors["physical_y"],
        event_idx=event_idx,
    )


def count_classes(graphs):
    counter = Counter()
    for graph in graphs:
        counter.update(graph.physical_y.tolist())
    return dict(sorted(counter.items()))


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-file",
        default=project_root / "data/28apr_00/events.root",
        type=Path,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--keep-noise", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root_file = args.root_file.resolve()
    model_path = project_root / "models/simple_3_class_tiny_gnn.pt"
    pred_plot_path = project_root / "figures/simple_3_class_event9_predicted.png"
    truth_plot_path = project_root / "figures/simple_3_class_event9_truth.png"

    print(f"Reading ROOT file: {root_file}")
    events = read_ecal_rechits_with_truth(root_file, max_events=10)
    if len(events) != 10:
        raise ValueError(f"Expected exactly 10 events, found {len(events)}.")

    filter_noise = not args.keep_noise
    print(
        "Noise handling: "
        + ("filtering out noise hits before training/evaluation" if filter_noise else "keeping noise hits")
    )

    graphs = []
    for event_idx, event in enumerate(events):
        n_noise = sum(bool(v) for v in event["noise_flag"])
        print(f"event {event_idx}: raw ECal hits={len(event['x'])}, noise_hits={n_noise}")
        graph = event_to_graph(event, event_idx=event_idx, filter_noise=filter_noise)
        graphs.append(graph)
        print(
            f"event {event_idx}: selected_hits={graph.num_nodes}, "
            f"labels={sorted(set(graph.physical_y.tolist()))}"
        )

    train_graphs = graphs[:9]
    test_graph = graphs[9]

    unique_labels = sorted({label for graph in graphs for label in graph.physical_y.tolist()})
    if unique_labels != list(VALID_LABELS):
        raise ValueError(
            f"Expected physical labels {VALID_LABELS}, but saw {unique_labels}. "
            "Check that origin_id_contribs contains the intended 3-class labels."
        )

    print(f"training events: 0-8 ({len(train_graphs)} events)")
    print("evaluation event: 9")
    print("training class counts:", count_classes(train_graphs))
    print("unique labels seen:", unique_labels)

    loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyNodeGNN(in_dim=train_graphs[0].x.shape[1], hidden_dim=args.hidden_dim, out_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_nodes = 0
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

        print(f"epoch={epoch:03d} train_loss={total_loss / total_nodes:.4f}")

    model.eval()
    with torch.no_grad():
        test_graph = test_graph.to(device)
        logits = model(test_graph.x, test_graph.edge_index)
        pred_class = logits.argmax(dim=1).cpu()
        true_class = test_graph.y.cpu()
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
        },
        model_path,
    )
    print(f"saved model: {model_path}")

    pos = test_graph.pos.cpu()
    plot_ecal_hit_classes_3d(
        pos,
        pred_physical,
        pred_plot_path,
        "Event 9 ECal hits, predicted dominant origin_id",
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
