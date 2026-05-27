"""Build an ML-ready ECal/TriggerPad cache with one tensor shard per ROOT file."""

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_shards import (
    ShardedECalTpadDataset,
    prepare_sharded_tensor_cache,
    validate_sharded_tensor_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--electron-count", type=int, default=None)
    parser.add_argument("--source-label", default="input")
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("ELECTRON_COUNT", "LABEL", "ROOT_DIR"),
        help="Repeat for a mixed cache, e.g. --source 2 2e data/.../2e/events --source 3 3e data/.../3e/events.",
    )
    parser.add_argument("--valid-labels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-root-files", type=int, default=None, help="Smoke-only limit applied per source.")
    parser.add_argument("--max-events-per-root-file", type=int, default=None, help="Smoke-only event limit per shard.")
    parser.add_argument(
        "--resume-from-root-index",
        type=int,
        default=1,
        help="Trust indexed sources before this 1-based ROOT-file position and resume work here.",
    )
    parser.add_argument("--read-step-size", type=int, default=500)
    parser.add_argument(
        "--filter-noise",
        action="store_true",
        help="Discard noise hits in the stored shards. By default shards retain explicit noise targets for later training-time policy.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Reuse already valid shard files while completing missing shards.")
    parser.add_argument(
        "--skip-failed-root-files",
        action="store_true",
        help="Record ROOT files that cannot be read/tensorized in index.json and continue with later sources.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild selected shard files even when a cache exists.")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.exists() or path.is_absolute():
        return path
    for root in (PROJECT_ROOT, PROJECT_ROOT.parent):
        candidate = root / path
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def root_specs_from_args(args):
    if args.source:
        if args.input_root_dir is not None:
            raise ValueError("Use either --input-root-dir or repeatable --source, not both.")
        return [
            (int(electron_count), label, resolve_path(root_dir))
            for electron_count, label, root_dir in args.source
        ]
    if args.input_root_dir is None:
        raise ValueError("Provide --input-root-dir or at least one --source.")
    return [(args.electron_count, args.source_label, resolve_path(args.input_root_dir))]


def main():
    args = parse_args()
    if args.read_step_size < 0:
        raise ValueError("--read-step-size must be non-negative.")
    if args.resume_from_root_index < 1:
        raise ValueError("--resume-from-root-index must be at least 1.")
    root_specs = root_specs_from_args(args)
    output_dir = resolve_path(args.output_dir)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("preprocess_ecal_tpad_sharded")
    read_step_size = args.read_step_size if args.read_step_size > 0 else None
    store_noise_targets = not args.filter_noise
    logger.info(
        "Stored noise policy: %s",
        "retain flagged hits with explicit background targets" if store_noise_targets else "discard flagged hits",
    )
    prepare_sharded_tensor_cache(
        output_dir,
        root_specs=root_specs,
        valid_labels=tuple(args.valid_labels),
        filter_noise=args.filter_noise,
        supervise_noise=store_noise_targets,
        force=args.force,
        skip_existing=args.skip_existing or not args.force,
        max_root_files=args.max_root_files,
        max_events_per_root_file=args.max_events_per_root_file,
        read_step_size=read_step_size,
        skip_failed_root_files=args.skip_failed_root_files,
        resume_from_root_index=args.resume_from_root_index,
        logger=logger,
    )
    _manifest, index = validate_sharded_tensor_cache(output_dir, load_shards=True)
    dataset = ShardedECalTpadDataset(output_dir)
    print(f"cache: {output_dir}")
    print(f"shards: {len(index['shards'])}; events: {len(dataset)}")
    print("source ROOT files:")
    for shard in index["shards"]:
        print(f"  {Path(shard['source']['path']).name}: {shard['num_events']} event(s) -> {shard['path']}")
    skipped_sources = index.get("skipped_sources", [])
    if skipped_sources:
        print("skipped ROOT files:")
        for skipped in skipped_sources:
            print(
                f"  {Path(skipped['source']['path']).name}: "
                f"{skipped.get('error_type', 'error')}: {skipped.get('error', '')}"
            )


if __name__ == "__main__":
    main()
