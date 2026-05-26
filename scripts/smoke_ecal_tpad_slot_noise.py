"""CPU smoke validation for explicit slot-model noise/background supervision."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    load_grouped_root_tensor_events,
)
from mldmx.eval.ecal_tpad_slot_model import evaluate
from mldmx.models import ECalTpadSlotModel
from mldmx.train.ecal_tpad_slot_model import compute_event_losses


VALID_LABELS = (1, 2, 3)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/ldmx_overlay_events_700k"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate explicit ECalTpadSlotModel noise targets from ROOT inputs on CPU."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--electron-count", type=int, default=3)
    parser.add_argument("--events-to-scan", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def canonical_order_from_non_noise_hits(event):
    noise_mask = event["is_noise_target"]
    origin_ids = event["origin_id_y"]
    positions = event["ecal_pos"]
    means = []
    for origin_id in sorted({int(value) for value in origin_ids[~noise_mask].tolist()}):
        mask = (~noise_mask) & (origin_ids == origin_id)
        means.append((origin_id, float(positions[mask, 1].mean().item())))
    return [origin_id for origin_id, _mean in sorted(means, key=lambda item: (item[1], item[0]))]


def main():
    args = parse_args()
    if args.events_to_scan <= 0:
        raise ValueError("--events-to-scan must be positive.")
    torch.manual_seed(args.seed)

    source_dir = args.data_root / f"{args.electron_count}e/events"
    events, _sources, _files = load_grouped_root_tensor_events(
        root_specs=[(args.electron_count, f"{args.electron_count}e", source_dir)],
        events_per_source=args.events_to_scan,
        valid_labels=VALID_LABELS,
        filter_noise=False,
        supervise_noise=True,
        allow_fewer_events=False,
        disable_progress=True,
        read_step_size=50,
    )
    noisy_events = [event for event in events if bool(event["is_noise_target"].any().item())]
    if not noisy_events:
        raise AssertionError(
            f"No flagged noise hit found in the first {args.events_to_scan} event(s) from {source_dir}."
        )

    event = noisy_events[0]
    apply_variable_count_target_mode(
        event,
        valid_labels=VALID_LABELS,
        target_mode="canonical-y",
        max_electrons=len(VALID_LABELS),
    )
    noise_mask = event["is_noise_target"]
    if not bool((event["physical_y"][noise_mask] == 0).all().item()):
        raise AssertionError("Noise hits were not assigned slot-model background class 0.")
    if not bool((event["canonical_y"][noise_mask] == -1).all().item()):
        raise AssertionError("Noise hits unexpectedly received electron canonical-y labels.")
    expected_order = canonical_order_from_non_noise_hits(event)
    if event["target_label_order"] != expected_order:
        raise AssertionError("Canonical-y ordering was changed by retained noise hits.")

    model = ECalTpadSlotModel(
        in_dim=int(event["x"].shape[1]),
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        max_electrons=len(VALID_LABELS),
        dropout=0.0,
        use_type_embedding=True,
    ).cpu()
    loss_args = SimpleNamespace(
        lambda_origin=1.0,
        lambda_fraction=1.0,
        lambda_slot=0.5,
        lambda_count=1.0,
        origin_class_weights=None,
        count_class_weights=None,
        batch_size=1,
        max_electrons=len(VALID_LABELS),
    )
    losses = compute_event_losses(model, event, torch.device("cpu"), loss_args)
    expected_noise_fraction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    if not torch.equal(losses["fraction_target"][noise_mask].cpu(), expected_noise_fraction.unsqueeze(0).expand(int(noise_mask.sum()), -1)):
        raise AssertionError("Noise fraction targets are not background-only one-hot rows.")
    for name in ("total_loss", "origin_loss", "fraction_loss", "slot_loss", "count_loss"):
        if not bool(torch.isfinite(losses[name]).item()):
            raise AssertionError(f"{name} is not finite.")
    losses["total_loss"].backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Backward produced no parameter gradients.")
    metrics, _predictions = evaluate(
        model,
        [event],
        [0],
        loss_args,
        torch.device("cpu"),
        "noise",
    )
    background_rows = sum(metrics["noise_hit_confusion"][0])
    if background_rows != int(noise_mask.sum().item()):
        raise AssertionError("Evaluation confusion matrix did not retain the background truth row.")

    print(f"source: {source_dir}")
    print(f"input shape: {tuple(event['x'].shape)}")
    print(f"noise hits supervised: {int(noise_mask.sum().item())}")
    print(f"noise hard targets: {event['physical_y'][noise_mask].tolist()}")
    print(f"noise canonical-y sentinel: {event['canonical_y'][noise_mask].tolist()}")
    print(f"canonical electron order: {event['target_label_order']}")
    print(f"count target: {int(losses['count_target'].item())}")
    print(f"background fraction target: {losses['fraction_target'][noise_mask][0].tolist()}")
    print(f"evaluated background truth hits: {background_rows}")
    print(f"total_loss: {float(losses['total_loss'].detach().item()):.6f}")
    print("status: pass (finite multi-task losses and backward)")


if __name__ == "__main__":
    main()
