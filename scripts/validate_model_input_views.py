"""Validate maintained-model input views from one saved canonical smoke event."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    load_processed_tensor_events,
)
from ml_ldmx.datasets.model_views import (
    ecal_gravnet_view,
    ecal_tpad_gravnet_view,
    ecal_tpad_slot_model_view,
    ecal_tpad_transformer_view,
    ecal_transformer_view,
)


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ecal_tpad_3class_smoke"
VALID_LABELS = (1, 2, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive all maintained model input views from one processed ECal/TPAD event."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--target-mode", default="canonical-y", choices=("canonical-y", "physical-origin"))
    parser.add_argument("--max-electrons", type=int, default=3)
    return parser.parse_args()


def _shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else "-"


def main():
    args = parse_args()
    if args.event_index < 0:
        raise ValueError("--event-index must be non-negative.")
    events, sources = load_processed_tensor_events(
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
        target_mode=args.target_mode,
        max_electrons=args.max_electrons,
    )
    builders = [
        ("ECalTransformer", ecal_transformer_view),
        ("ECalTpadTransformer", ecal_tpad_transformer_view),
        ("ECalGravNet", ecal_gravnet_view),
        ("ECalTpadGravNet", ecal_tpad_gravnet_view),
        ("ECalTpadSlotModel", ecal_tpad_slot_model_view),
    ]
    views = [(name, builder(event)) for name, builder in builders]
    if views[-1][1] is not event:
        raise AssertionError("ECalTpadSlotModel adapter must return the unchanged canonical event.")

    print(f"processed_dir: {args.processed_dir}")
    print(f"event_index: {args.event_index} source: {sources[args.event_index]}")
    print(f"target_mode: {args.target_mode} target_label_order: {event.get('target_label_order')}")
    print(
        f"{'model':<22} {'x':<12} {'ecal_mask':<12} {'tpad_mask':<12} "
        f"{'ecal_pos':<12} {'y':<12} {'origin_id_y':<12} {'canonical_y':<12}"
    )
    for name, view in views:
        print(
            f"{name:<22} {str(_shape(view['x'])):<12} {str(_shape(view.get('ecal_mask'))):<12} "
            f"{str(_shape(view.get('tpad_mask'))):<12} {str(_shape(view['ecal_pos'])):<12} "
            f"{str(_shape(view['y'])):<12} {str(_shape(view['origin_id_y'])):<12} "
            f"{str(_shape(view.get('canonical_y'))):<12}"
        )


if __name__ == "__main__":
    main()
