"""CPU smoke validation for the maintained GravNetConv baseline models."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    load_processed_tensor_events,
)
from mldmx.datasets.model_views import ecal_gravnet_view, ecal_tpad_gravnet_view
from mldmx.models import ECalGravNet, ECalTpadGravNet


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ecal_tpad_3class_smoke"
VALID_LABELS = (1, 2, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run forward/backward CPU smoke checks for maintained GravNet baselines."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def run_model_check(name, model_cls, view, args):
    model = model_cls(
        in_dim=int(view["x"].shape[1]),
        hidden_dim=args.hidden_dim,
        out_dim=len(VALID_LABELS),
        num_layers=args.num_layers,
        space_dimensions=4,
        propagate_dimensions=16,
        k=args.k,
        dropout=0.0,
    ).cpu()
    logits = model(view["x"].to(dtype=torch.float32))
    supervised_logits = logits[view["ecal_mask"]]
    target = view["y"].to(dtype=torch.long)
    if supervised_logits.shape[0] != target.shape[0]:
        raise AssertionError(
            f"{name}: supervised logits and targets differ: "
            f"{supervised_logits.shape[0]} vs {target.shape[0]}."
        )
    loss = F.cross_entropy(supervised_logits, target)
    if not bool(torch.isfinite(loss).item()):
        raise AssertionError(f"{name}: loss is not finite: {float(loss.detach())}.")
    loss.backward()
    if not any(param.grad is not None for param in model.parameters()):
        raise AssertionError(f"{name}: backward produced no parameter gradients.")
    return {
        "input_shape": tuple(view["x"].shape),
        "logits_shape": tuple(logits.shape),
        "supervised_nodes": int(view["ecal_mask"].sum().item()),
        "loss": float(loss.detach().cpu().item()),
    }


def main():
    args = parse_args()
    if args.event_index < 0:
        raise ValueError("--event-index must be non-negative.")
    torch.manual_seed(args.seed)

    events, _sources = load_processed_tensor_events(
        args.processed_dir,
        max_events=args.event_index + 1,
    )
    if args.event_index >= len(events):
        raise ValueError(
            f"Requested event index {args.event_index}, but only loaded {len(events)} event(s)."
        )
    event = events[args.event_index]
    apply_variable_count_target_mode(
        event,
        valid_labels=VALID_LABELS,
        target_mode="canonical-y",
        max_electrons=len(VALID_LABELS),
    )

    checks = [
        ("ECalGravNet", ECalGravNet, ecal_gravnet_view(event)),
        ("ECalTpadGravNet", ECalTpadGravNet, ecal_tpad_gravnet_view(event)),
    ]
    print(f"target_mode: canonical-y target_label_order: {event['target_label_order']}")
    print(f"{'model':<18} {'input':<12} {'logits':<12} {'supervised':<12} {'loss':<10} status")
    for name, model_cls, view in checks:
        metrics = run_model_check(name, model_cls, view, args)
        print(
            f"{name:<18} {str(metrics['input_shape']):<12} {str(metrics['logits_shape']):<12} "
            f"{metrics['supervised_nodes']:<12} {metrics['loss']:<10.6f} pass"
        )


if __name__ == "__main__":
    main()
