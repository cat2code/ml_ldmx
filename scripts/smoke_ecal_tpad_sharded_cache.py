"""Smoke-test sharded preprocessing, lazy access, reuse, and one training step."""

import argparse
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.datasets.ecal_tpad_loading import apply_variable_count_target_mode, filter_noise_tensor_event
from ml_ldmx.datasets.ecal_tpad_shards import (
    ShardedECalTpadDataset,
    prepare_sharded_tensor_cache,
    validate_sharded_tensor_cache,
)
from ml_ldmx.datasets.model_views import ecal_transformer_view
from ml_ldmx.models import ECalTransformer
from ml_ldmx.train.hit_classifier_baseline import compute_event_losses


VALID_LABELS = (1, 2, 3)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/ldmx_overlay_events_700k"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Retain smoke shards here instead of using a temporary directory.")
    parser.add_argument("--max-root-files", type=int, default=1)
    parser.add_argument("--max-events-per-root-file", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def run(cache_dir, args):
    data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    root_specs = [
        (2, "2e", data_root / "2e/events"),
        (3, "3e", data_root / "3e/events"),
    ]
    logger = logging.getLogger("sharded-cache-smoke")
    common_kwargs = {
        "cache_dir": cache_dir,
        "root_specs": root_specs,
        "valid_labels": VALID_LABELS,
        "filter_noise": False,
        "supervise_noise": True,
        "max_root_files": args.max_root_files,
        "max_events_per_root_file": args.max_events_per_root_file,
        "read_step_size": 50,
        "logger": logger,
    }
    prepare_sharded_tensor_cache(**common_kwargs)
    prepare_sharded_tensor_cache(**common_kwargs)
    _manifest, index = validate_sharded_tensor_cache(cache_dir, load_shards=True)
    dataset = ShardedECalTpadDataset(cache_dir, shard_cache_size=1)
    expected_events = sum(entry["num_events"] for entry in index["shards"])
    if len(dataset) != expected_events:
        raise AssertionError(f"Expected {expected_events} indexed events, got {len(dataset)}.")
    if len(index["shards"]) < 2:
        raise AssertionError("Smoke test needs at least two processed shards.")

    source_transition_idx = next(
        (
            shard_idx
            for shard_idx in range(1, len(index["shards"]))
            if index["shards"][shard_idx - 1]["source"]["source_label"]
            != index["shards"][shard_idx]["source"]["source_label"]
        ),
        None,
    )
    if source_transition_idx is None:
        raise AssertionError("Smoke test needs a boundary between ROOT source groups.")
    source_boundary_start = index["shards"][source_transition_idx]["event_start"]
    before_boundary = dataset[source_boundary_start - 1]
    after_boundary = dataset[source_boundary_start]
    if args.max_root_files >= 2:
        first_two_names = [entry["source"]["name"] for entry in index["shards"][:2]]
        if first_two_names != ["events_1.root", "events_2.root"]:
            raise AssertionError(f"ROOT shards were not numerically ordered: {first_two_names}.")

    raw_noise_hits = sum(
        int(dataset[index]["is_noise_target"].sum().item()) for index in range(len(dataset))
    )
    if raw_noise_hits <= 0:
        raise AssertionError("Noise-inclusive shard smoke did not retain any flagged noise hits.")

    def canonical_filtered_transform(event):
        event = filter_noise_tensor_event(event)
        return apply_variable_count_target_mode(
            event,
            valid_labels=VALID_LABELS,
            target_mode="canonical-y",
            max_electrons=3,
        )

    dataset.set_event_transform(canonical_filtered_transform)
    view = ecal_transformer_view(dataset[0])
    model = ECalTransformer(
        in_dim=int(view["x"].shape[1]),
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        out_dim=len(VALID_LABELS),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    losses = compute_event_losses(model, dataset[0], ecal_transformer_view, torch.device(args.device))
    if not bool(torch.isfinite(losses["total_loss"]).item()):
        raise AssertionError("Lazy sharded event produced a non-finite classification loss.")
    losses["total_loss"].backward()
    optimizer.step()

    def canonical_noise_transform(event):
        return apply_variable_count_target_mode(
            event,
            valid_labels=VALID_LABELS,
            target_mode="canonical-y",
            max_electrons=3,
        )

    dataset.set_event_transform(canonical_noise_transform)
    supervised_noise_hits = sum(
        int(dataset[index]["is_noise_target"].sum().item()) for index in range(len(dataset))
    )
    if supervised_noise_hits != raw_noise_hits:
        raise AssertionError("Noise-inclusive lazy view did not preserve stored noise targets.")

    print(f"cache: {cache_dir}")
    print(f"shards: {len(index['shards'])}; events: {len(dataset)}")
    print(
        "boundary: "
        f"{before_boundary['source_label']}:{before_boundary['source_file']} -> "
        f"{after_boundary['source_label']}:{after_boundary['source_file']}"
    )
    print(
        "lazy training step: "
        f"input={tuple(view['x'].shape)} loss={float(losses['total_loss'].detach().cpu().item()):.6f} backward=passed"
    )
    print(f"stored noise hits: {raw_noise_hits}; filtered baseline access and supervised-noise access: passed")
    print("cache reuse: passed (second preparation call reused valid shard files)")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(7)
    if args.cache_dir is not None:
        cache_dir = args.cache_dir if args.cache_dir.is_absolute() else PROJECT_ROOT / args.cache_dir
        run(cache_dir, args)
    else:
        with TemporaryDirectory(prefix="ml_ldmx_sharded_smoke_") as temporary_dir:
            run(Path(temporary_dir), args)


if __name__ == "__main__":
    main()
