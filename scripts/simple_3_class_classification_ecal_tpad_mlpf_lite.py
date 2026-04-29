"""
Run from the mldmx directory:

    cd mldmx
    python3 -m pip install -e .
    python3 scripts/simple_3_class_classification_ecal_tpad_mlpf_lite.py
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from mldmx.datasets.tensorize import (
    origin_energy_fraction_targets,
    tensorize_ecal_with_triggerpad_context,
)
from mldmx.io.root_reader import read_ecal_rechits_with_truth_and_triggerpad_context
from mldmx.models import ECalTpadMLPFLiteTransformer
from mldmx.train.losses import soft_label_cross_entropy
from mldmx.train.utils import choose_device
from mldmx.viz.ecal import plot_ecal_hit_classes_3d
from mldmx.viz.fractions import (
    plot_fraction_error_3d,
    plot_fraction_mae_hist,
    plot_fraction_purity,
    plot_fraction_scatter,
)


VALID_LABELS = (1, 2, 3)
FRACTION_SUM_WARNING_ATOL = 1e-3


def warn_fraction_target_sums(event, event_idx, fraction_target, keep_indices):
    row_sums = fraction_target.sum(dim=1)
    bad_rows = torch.nonzero(
        (row_sums - 1.0).abs() > FRACTION_SUM_WARNING_ATOL,
        as_tuple=False,
    ).flatten()
    if bad_rows.numel() == 0:
        return

    hit_ids = event.get("hit_id", list(range(len(event["edep_contribs"]))))
    print(
        f"WARNING event {event_idx}: {bad_rows.numel()} fraction target rows "
        f"sum outside 1 +/- {FRACTION_SUM_WARNING_ATOL}"
    )
    for row_idx in bad_rows[:10].tolist():
        original_hit_idx = int(keep_indices[row_idx])
        print(
            "  "
            f"event={event_idx} row={row_idx} original_hit_index={original_hit_idx} "
            f"hit_id={hit_ids[original_hit_idx]} row_sum={row_sums[row_idx].item():.6f}"
        )
    if bad_rows.numel() > 10:
        print(f"  ... omitted {bad_rows.numel() - 10} additional row-sum warnings")


def event_to_tensors(event, event_idx, filter_noise=True):
    tensors = tensorize_ecal_with_triggerpad_context(
        event,
        valid_labels=VALID_LABELS,
        filter_noise=filter_noise,
    )
    fraction_target = origin_energy_fraction_targets(
        event,
        keep_indices=tensors["keep_indices"],
        valid_labels=VALID_LABELS,
    )
    warn_fraction_target_sums(
        event,
        event_idx=event_idx,
        fraction_target=fraction_target,
        keep_indices=tensors["keep_indices"],
    )
    tensors["fraction_target"] = fraction_target
    tensors["event_idx"] = event_idx
    return tensors


def count_classes(events):
    counter = Counter()
    for event in events:
        counter.update(event["physical_y"].tolist())
    return dict(sorted(counter.items()))


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-file",
        default=project_root / "data/28apr_00/events.root",
        type=Path,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-fraction", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help="Use CPU by default for reproducible smoke tests; pass auto to use CUDA/MPS if available.",
    )
    parser.add_argument("--keep-noise", action="store_true")
    return parser.parse_args()


def load_tensor_events(args):
    root_file = args.root_file.resolve()
    print(f"Reading ROOT file: {root_file}")
    events = read_ecal_rechits_with_truth_and_triggerpad_context(root_file, max_events=10)
    print(f"number of events: {len(events)}")

    filter_noise = not args.keep_noise
    tensor_events = []
    for event_idx, event in enumerate(events):
        n_noise = sum(bool(v) for v in event["noise_flag"])
        n_tpad = len(event["trigger_pad_tracks"]["centroid"])
        print(
            f"event {event_idx}: raw ECal hits={len(event['x'])}, "
            f"noise_hits={n_noise}, TriggerPadTracks={n_tpad}"
        )
        tensor_event = event_to_tensors(
            event,
            event_idx=event_idx,
            filter_noise=filter_noise,
        )
        tensor_events.append(tensor_event)
        print(
            f"event {event_idx}: selected_ecal_hits={tensor_event['ecal_mask'].sum().item()}, "
            f"tpad_nodes={tensor_event['tpad_mask'].sum().item()}, "
            f"labels={sorted(set(tensor_event['physical_y'].tolist()))}"
        )

    return tensor_events


def main():
    args = parse_args()
    torch.manual_seed(7)

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models/simple_3_class_ecal_tpad_mlpf_lite.pt"
    pred_plot_path = project_root / "figures/simple_3_class_event9_tpad_mlpf_lite_predicted.png"
    truth_plot_path = project_root / "figures/simple_3_class_event9_tpad_mlpf_lite_truth.png"
    frac_error_path = project_root / "figures/simple_3_class_event9_tpad_mlpf_lite_fraction_error.png"
    frac_scatter_path = project_root / "figures/simple_3_class_event9_tpad_mlpf_lite_fraction_scatter.png"
    frac_purity_path = project_root / "figures/simple_3_class_event9_tpad_mlpf_lite_fraction_purity.png"
    frac_mae_hist_path = project_root / "figures/simple_3_class_event9_tpad_mlpf_lite_fraction_mae_hist.png"

    filter_noise = not args.keep_noise
    print(
        "Noise handling: "
        + ("filtering out noise hits before training/evaluation" if filter_noise else "keeping noise hits")
    )

    tensor_events = load_tensor_events(args)
    if len(tensor_events) < 10:
        raise ValueError(f"Expected at least 10 events, found {len(tensor_events)}.")

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
    print(f"lambda_fraction: {args.lambda_fraction}")
    model = ECalTpadMLPFLiteTransformer(
        input_dim=train_events[0]["x"].shape[1],
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
        total_loss_sum = 0.0
        origin_loss_sum = 0.0
        fraction_loss_sum = 0.0
        total_ecal_hits = 0
        for event in train_events:
            x = event["x"].to(device)
            ecal_mask = event["ecal_mask"].to(device)
            y = event["y"].to(device)
            fraction_target = event["fraction_target"].to(device)

            outputs = model(x)
            ecal_origin_logits = outputs["origin_logits"][ecal_mask]
            ecal_fraction_logits = outputs["fraction_logits"][ecal_mask]
            origin_loss = F.cross_entropy(ecal_origin_logits, y)
            fraction_loss = soft_label_cross_entropy(ecal_fraction_logits, fraction_target)
            loss = origin_loss + args.lambda_fraction * fraction_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_hits = y.numel()
            total_loss_sum += loss.item() * num_hits
            origin_loss_sum += origin_loss.item() * num_hits
            fraction_loss_sum += fraction_loss.item() * num_hits
            total_ecal_hits += num_hits

        print(
            f"epoch={epoch:03d} "
            f"train_loss={total_loss_sum / total_ecal_hits:.4f} "
            f"origin_ce={origin_loss_sum / total_ecal_hits:.4f} "
            f"fraction_ce={fraction_loss_sum / total_ecal_hits:.4f}"
        )

    model.eval()
    with torch.no_grad():
        x = test_event["x"].to(device)
        ecal_mask = test_event["ecal_mask"].to(device)
        outputs = model(x)
        ecal_origin_logits = outputs["origin_logits"][ecal_mask]
        ecal_fraction_logits = outputs["fraction_logits"][ecal_mask]
        ecal_fraction_pred = outputs["fraction_pred"][ecal_mask]

        pred_class = ecal_origin_logits.argmax(dim=1).cpu()
        true_class = test_event["y"].cpu()
        accuracy = (pred_class == true_class).float().mean().item()

        fraction_target = test_event["fraction_target"].to(device)
        fraction_ce = soft_label_cross_entropy(ecal_fraction_logits, fraction_target).item()
        fraction_mse = F.mse_loss(ecal_fraction_pred, fraction_target).item()
        fraction_abs_error = (ecal_fraction_pred - fraction_target).abs()
        fraction_mae = fraction_abs_error.mean().item()
        per_hit_fraction_mae = fraction_abs_error.mean(dim=1).cpu()
        mean_max_pred_fraction = ecal_fraction_pred.max(dim=1).values.mean().item()
        mean_max_target_fraction = fraction_target.max(dim=1).values.mean().item()

    class_to_label = {0: 1, 1: 2, 2: 3}
    pred_physical = torch.tensor([class_to_label[int(v)] for v in pred_class], dtype=torch.long)
    true_physical = torch.tensor([class_to_label[int(v)] for v in true_class], dtype=torch.long)
    print(f"event 9 origin accuracy: {accuracy:.3f}")
    print(f"event 9 fraction soft-label CE: {fraction_ce:.4f}")
    print(f"event 9 fraction MSE: {fraction_mse:.4f}")
    print(f"event 9 fraction MAE: {fraction_mae:.4f}")
    print(f"event 9 mean max predicted fraction: {mean_max_pred_fraction:.4f}")
    print(f"event 9 mean max target fraction: {mean_max_target_fraction:.4f}")
    print("event 9 true class counts:", dict(sorted(Counter(true_physical.tolist()).items())))
    print("event 9 predicted class counts:", dict(sorted(Counter(pred_physical.tolist()).items())))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "valid_labels": VALID_LABELS,
            "class_to_label": class_to_label,
            "model_kwargs": {
                "input_dim": train_events[0]["x"].shape[1],
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "out_dim": 3,
            },
            "lambda_fraction": args.lambda_fraction,
        },
        model_path,
    )
    print(f"saved model: {model_path}")

    pos = test_event["ecal_pos"].cpu()
    fraction_target_cpu = test_event["fraction_target"].cpu()
    fraction_pred_cpu = ecal_fraction_pred.cpu()

    plot_ecal_hit_classes_3d(
        pos,
        pred_physical,
        pred_plot_path,
        "Event 9 ECal hits, MLPF-lite transformer predicted dominant origin_id",
    )
    plot_ecal_hit_classes_3d(
        pos,
        true_physical,
        truth_plot_path,
        "Event 9 ECal hits, true dominant origin_id",
    )
    plot_fraction_error_3d(pos, per_hit_fraction_mae, frac_error_path)
    plot_fraction_scatter(
        fraction_target_cpu,
        fraction_pred_cpu,
        frac_scatter_path,
        valid_labels=VALID_LABELS,
    )
    plot_fraction_purity(fraction_target_cpu, fraction_pred_cpu, frac_purity_path)
    plot_fraction_mae_hist(per_hit_fraction_mae, frac_mae_hist_path)

    print(f"saved prediction plot: {pred_plot_path}")
    print(f"saved truth plot: {truth_plot_path}")
    print(f"saved fraction error plot: {frac_error_path}")
    print(f"saved fraction scatter plot: {frac_scatter_path}")
    print(f"saved fraction purity plot: {frac_purity_path}")
    print(f"saved fraction MAE histogram: {frac_mae_hist_path}")


if __name__ == "__main__":
    main()
