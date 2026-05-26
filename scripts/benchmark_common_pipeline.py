"""Lightweight throughput benchmark for the maintained common event pipeline."""

import argparse
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode_to_events,
    load_ecal_tpad_tensor_events,
    load_processed_tensor_events,
)
from mldmx.datasets.model_views import (
    ecal_gravnet_view,
    ecal_tpad_gravnet_view,
    ecal_tpad_slot_model_view,
    ecal_tpad_transformer_view,
    ecal_transformer_view,
)
from mldmx.io.root_files import find_root_files
from mldmx.models import (
    ECalGravNet,
    ECalTpadGravNet,
    ECalTpadSlotModel,
    ECalTpadTransformer,
    ECalTransformer,
)
from mldmx.train.ecal_tpad_slot_model import compute_event_losses as compute_slot_event_losses


VALID_LABELS = (1, 2, 3)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ecal_tpad_3class_smoke"
DEFAULT_ROOT_DIR = PROJECT_ROOT / "data/ldmx_overlay_events_700k/3e/events"
VIEW_FUNCTIONS = {
    "ECalGravNet": ecal_gravnet_view,
    "ECalTpadGravNet": ecal_tpad_gravnet_view,
    "ECalTransformer": ecal_transformer_view,
    "ECalTpadTransformer": ecal_tpad_transformer_view,
    "ECalTpadSlotModel": ecal_tpad_slot_model_view,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark shared loading, views, and maintained model steps.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--read-step-size", type=int, default=50)
    parser.add_argument("--view-iterations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--skip-root", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    return parser.parse_args()


def elapsed_seconds(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare_canonical_events(events):
    apply_variable_count_target_mode_to_events(
        events,
        valid_labels=VALID_LABELS,
        target_mode="canonical-y",
        max_electrons=len(VALID_LABELS),
    )
    return events


def resolve_device(device_name):
    if device_name in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_for_name(name, input_dim):
    if name == "ECalGravNet":
        return ECalGravNet(
            in_dim=input_dim,
            hidden_dim=32,
            out_dim=3,
            num_layers=1,
            space_dimensions=4,
            propagate_dimensions=16,
            k=8,
            dropout=0.0,
        )
    if name == "ECalTpadGravNet":
        return ECalTpadGravNet(
            in_dim=input_dim,
            hidden_dim=32,
            out_dim=3,
            num_layers=1,
            space_dimensions=4,
            propagate_dimensions=16,
            k=8,
            dropout=0.0,
        )
    if name == "ECalTransformer":
        return ECalTransformer(
            in_dim=input_dim,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            out_dim=3,
        )
    if name == "ECalTpadTransformer":
        return ECalTpadTransformer(
            in_dim=input_dim,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            out_dim=3,
        )
    return ECalTpadSlotModel(
        in_dim=input_dim,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        max_electrons=3,
        dropout=0.0,
        use_type_embedding=True,
    )


def baseline_loss(model, view, device):
    x = view["x"].to(device=device, dtype=torch.float32)
    mask = view["ecal_mask"].to(device=device, dtype=torch.bool)
    target = view["y"].to(device=device, dtype=torch.long)
    logits = model(x)
    return F.cross_entropy(logits[mask], target)


def benchmark_model_steps(name, events, view_fn, device, steps, use_cached_views):
    cached_views = [view_fn(event) for event in events] if use_cached_views else None
    example = cached_views[0] if cached_views is not None else view_fn(events[0])
    model = model_for_name(name, int(example["x"].shape[1])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    slot_args = SimpleNamespace(
        lambda_origin=1.0,
        lambda_fraction=1.0,
        lambda_slot=0.5,
        lambda_count=1.0,
        origin_class_weights=None,
        count_class_weights=None,
    )

    def one_step(step_idx):
        optimizer.zero_grad(set_to_none=True)
        event_idx = step_idx % len(events)
        if name == "ECalTpadSlotModel":
            event = cached_views[event_idx] if cached_views is not None else view_fn(events[event_idx])
            loss = compute_slot_event_losses(model, event, device, slot_args)["total_loss"]
        else:
            view = cached_views[event_idx] if cached_views is not None else view_fn(events[event_idx])
            loss = baseline_loss(model, view, device)
        loss.backward()
        optimizer.step()

    one_step(0)
    synchronize(device)
    start = time.perf_counter()
    for step_idx in range(steps):
        one_step(step_idx)
    synchronize(device)
    return (time.perf_counter() - start) / steps


def main():
    args = parse_args()
    if args.max_events <= 0 or args.view_iterations <= 0 or args.steps <= 0:
        raise ValueError("--max-events, --view-iterations, and --steps must be positive.")
    if args.read_step_size < 0:
        raise ValueError("--read-step-size must be non-negative.")
    torch.manual_seed(7)
    logger = logging.getLogger("pipeline-benchmark")
    logger.addHandler(logging.NullHandler())
    device = resolve_device(args.device)

    print(f"device: {device}")
    print(f"events requested: {args.max_events}; model steps: {args.steps}; view iterations: {args.view_iterations}")

    (processed_result, processed_elapsed) = elapsed_seconds(
        lambda: load_processed_tensor_events(args.processed_dir, max_events=args.max_events, logger=logger)
    )
    processed_events, _processed_sources = processed_result
    (_, canonical_elapsed) = elapsed_seconds(lambda: prepare_canonical_events(processed_events))
    print("\nloading")
    print(
        f"processed cache load        {processed_elapsed:.6f} s  "
        f"{len(processed_events) / processed_elapsed:8.2f} events/s"
    )
    print(
        f"canonical-y preparation     {canonical_elapsed:.6f} s  "
        f"{len(processed_events) / canonical_elapsed:8.2f} events/s"
    )

    if not args.skip_root:
        root_files = find_root_files(args.root_dir)
        read_step_size = args.read_step_size if args.read_step_size > 0 else None
        (root_result, root_elapsed) = elapsed_seconds(
            lambda: load_ecal_tpad_tensor_events(
                root_files=root_files,
                max_events=args.max_events,
                valid_labels=VALID_LABELS,
                target_mode="physical-origin",
                filter_noise=True,
                allow_fewer_events=True,
                data_dir=args.root_dir,
                logger=logger,
                disable_progress=True,
                read_step_size=read_step_size,
            )
        )
        root_events, _root_sources = root_result
        (_, root_canonical_elapsed) = elapsed_seconds(lambda: prepare_canonical_events(root_events))
        print(
            f"ROOT read + tensorize       {root_elapsed:.6f} s  "
            f"{len(root_events) / root_elapsed:8.2f} events/s  step_size={read_step_size}"
        )
        print(
            f"ROOT canonical-y prep       {root_canonical_elapsed:.6f} s  "
            f"{len(root_events) / root_canonical_elapsed:8.2f} events/s"
        )

    print("\nadapter/view preparation")
    print(f"{'view':<23} {'ms/call':>12} {'calls/s':>12}")
    for name, view_fn in VIEW_FUNCTIONS.items():
        calls = len(processed_events) * args.view_iterations
        (_, view_elapsed) = elapsed_seconds(
            lambda fn=view_fn: [
                fn(event)
                for _iteration in range(args.view_iterations)
                for event in processed_events
            ]
        )
        print(f"{name:<23} {view_elapsed * 1000 / calls:12.4f} {calls / view_elapsed:12.2f}")

    if args.skip_models:
        return

    print("\nforward/backward step")
    print(f"{'model':<23} {'on-demand ms':>14} {'cached-view ms':>15} {'speedup':>10}")
    for name, view_fn in VIEW_FUNCTIONS.items():
        try:
            on_demand = benchmark_model_steps(
                name, processed_events, view_fn, device, args.steps, use_cached_views=False
            )
            cached = benchmark_model_steps(
                name, processed_events, view_fn, device, args.steps, use_cached_views=True
            )
        except ImportError as exc:
            print(f"{name:<23} skipped: {exc}")
            continue
        print(f"{name:<23} {on_demand * 1000:14.3f} {cached * 1000:15.3f} {on_demand / cached:10.3f}x")


if __name__ == "__main__":
    main()
