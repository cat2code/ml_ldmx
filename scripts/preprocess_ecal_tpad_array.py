"""Plan, run, and finalize parallel ROOT-to-shard preprocessing."""

import argparse
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.datasets.ecal_tpad_shards import (
    create_parallel_shard_plan,
    finalize_parallel_shard_plan,
    load_parallel_shard_plan,
    prepare_parallel_shard_task,
)


def _source_specs(values):
    return [
        (int(electron_count), source_label, Path(root_dir).resolve())
        for electron_count, source_label, root_dir in values
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Freeze ordered ROOT files and shard paths.")
    plan_parser.add_argument("--plan-file", type=Path, required=True)
    plan_parser.add_argument("--output-root", type=Path, required=True)
    plan_parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        required=True,
        metavar=("ELECTRON_COUNT", "LABEL", "ROOT_DIR"),
    )
    plan_parser.add_argument("--valid-labels", type=int, nargs="+", default=[1, 2, 3])
    plan_parser.add_argument("--ecal-energy-transform", choices=("raw", "log1p"), default="log1p")
    plan_parser.add_argument("--tpad-pe-transform", choices=("raw", "log1p"), default="log1p")
    plan_parser.add_argument("--filter-noise", action="store_true")
    plan_parser.add_argument("--max-root-files", type=int)
    plan_parser.add_argument("--max-events-per-root-file", type=int)

    count_parser = subparsers.add_parser("count", help="Print the number of worker tasks.")
    count_parser.add_argument("--plan-file", type=Path, required=True)

    worker_parser = subparsers.add_parser("worker", help="Run one frozen worker task.")
    worker_parser.add_argument("--plan-file", type=Path, required=True)
    worker_parser.add_argument("--task-index", type=int, required=True)
    worker_parser.add_argument("--read-step-size", type=int, default=500)
    worker_parser.add_argument("--force", action="store_true")

    finalize_parser = subparsers.add_parser("finalize", help="Build and validate cache metadata.")
    finalize_parser.add_argument("--plan-file", type=Path, required=True)
    finalize_parser.add_argument("--expected-events-2e", type=int, default=0)
    finalize_parser.add_argument("--expected-events-3e", type=int, default=0)
    finalize_parser.add_argument("--allow-failed-root-files", action="store_true")
    finalize_parser.add_argument("--load-all-shards", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("preprocess_ecal_tpad_array")

    if args.command == "plan":
        plan = create_parallel_shard_plan(
            args.plan_file,
            output_root=args.output_root,
            root_specs=_source_specs(args.source),
            valid_labels=tuple(args.valid_labels),
            filter_noise=args.filter_noise,
            supervise_noise=not args.filter_noise,
            max_root_files=args.max_root_files,
            max_events_per_root_file=args.max_events_per_root_file,
            ecal_energy_transform=args.ecal_energy_transform,
            tpad_pe_transform=args.tpad_pe_transform,
        )
        print(f"plan: {args.plan_file.resolve()}")
        print(f"tasks: {len(plan['tasks'])}")
        for group in plan["source_groups"]:
            print(f"{group['source_label']}: {len(group['root_sources'])} ROOT file(s)")
        return

    if args.command == "count":
        print(len(load_parallel_shard_plan(args.plan_file)["tasks"]))
        return

    if args.command == "worker":
        if args.read_step_size < 0:
            raise ValueError("--read-step-size must be non-negative.")
        prepare_parallel_shard_task(
            args.plan_file,
            task_index=args.task_index,
            read_step_size=args.read_step_size or None,
            force=args.force,
            logger=logger,
        )
        return

    expected_events = {
        "2e": args.expected_events_2e,
        "3e": args.expected_events_3e,
    }
    summary = finalize_parallel_shard_plan(
        args.plan_file,
        expected_events_by_label=expected_events,
        allow_failed_root_files=args.allow_failed_root_files,
        load_all_shards=args.load_all_shards,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
