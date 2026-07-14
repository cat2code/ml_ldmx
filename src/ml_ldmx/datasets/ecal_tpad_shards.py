"""Scalable sharded storage for canonical ECal + TriggerPad tensor events."""

from bisect import bisect_right
from collections import OrderedDict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ml_ldmx.io.root_files import find_root_files
from ml_ldmx.io.root_reader import iter_ecal_rechits_with_truth_and_triggerpad_context
from ml_ldmx.datasets.tensorize import ECAL_ENERGY_TRANSFORMS, TPAD_PE_TRANSFORMS


SHARD_CACHE_SCHEMA_VERSION = 1
SHARD_PAYLOAD_SCHEMA_VERSION = 1
PARALLEL_PLAN_SCHEMA_VERSION = 1
FEATURE_LAYOUT = [
    "is_ecal",
    "is_tpad",
    "ecal_x",
    "ecal_y",
    "ecal_z",
    "ecal_energy",
    "tpad_centroid",
    "tpad_pe",
]


def _load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _root_file_metadata(path, electron_count, source_label, source_dir):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "source_dir": str(Path(source_dir).resolve()),
        "source_label": source_label,
        "electron_count": int(electron_count) if electron_count is not None else None,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _root_sources(root_specs, max_root_files=None):
    sources = []
    for electron_count, source_label, source_dir in root_specs:
        source_dir = Path(source_dir)
        root_files = find_root_files(source_dir)
        if max_root_files is not None:
            root_files = root_files[:max_root_files]
        sources.extend(
            _root_file_metadata(path, electron_count, source_label, source_dir)
            for path in root_files
        )
    return sources


def _cache_spec(
    root_sources,
    valid_labels,
    filter_noise,
    supervise_noise,
    max_events_per_root_file,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
):
    if ecal_energy_transform not in ECAL_ENERGY_TRANSFORMS:
        raise ValueError(
            f"Unknown ECal energy transform {ecal_energy_transform!r}; "
            f"expected one of {ECAL_ENERGY_TRANSFORMS}."
        )
    if tpad_pe_transform not in TPAD_PE_TRANSFORMS:
        raise ValueError(
            f"Unknown TriggerPadTracks pe transform {tpad_pe_transform!r}; "
            f"expected one of {TPAD_PE_TRANSFORMS}."
        )
    return {
        "reader": "ecal_tpad_sharded",
        "schema_version": SHARD_CACHE_SCHEMA_VERSION,
        "root_sources": root_sources,
        "valid_labels": list(valid_labels),
        "filter_noise": bool(filter_noise),
        "supervise_noise": bool(supervise_noise),
        "ecal_energy_transform": ecal_energy_transform,
        "tpad_pe_transform": tpad_pe_transform,
        "stored_target_mode": "physical-origin",
        "max_events_per_root_file": max_events_per_root_file,
        "feature_layout": FEATURE_LAYOUT,
    }


def _normalized_cache_spec(spec):
    spec = dict(spec or {})
    spec.setdefault("ecal_energy_transform", "raw")
    spec.setdefault("tpad_pe_transform", "raw")
    return spec


def _cache_specs_match(actual, requested):
    return _normalized_cache_spec(actual) == _normalized_cache_spec(requested)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def has_sharded_tensor_cache(cache_dir):
    cache_dir = Path(cache_dir)
    return (cache_dir / "manifest.json").exists() and (cache_dir / "index.json").exists()


def validate_sharded_cache_request(
    cache_dir,
    root_specs,
    valid_labels,
    filter_noise=True,
    supervise_noise=False,
    max_root_files=None,
    max_events_per_root_file=None,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
):
    """Require an existing cache to correspond to the requested raw dataset/settings."""
    manifest = _load_json(Path(cache_dir) / "manifest.json")
    requested_spec = _cache_spec(
        root_sources=_root_sources(root_specs, max_root_files=max_root_files),
        valid_labels=tuple(valid_labels),
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        max_events_per_root_file=max_events_per_root_file,
        ecal_energy_transform=ecal_energy_transform,
        tpad_pe_transform=tpad_pe_transform,
    )
    if not _cache_specs_match(manifest.get("cache_spec"), requested_spec):
        raise ValueError(
            f"Existing sharded cache does not match requested ROOT inputs/settings: {cache_dir}. "
            "Choose a different --processed-cache or pass --force-sharded-cache."
        )
    return requested_spec


def _validate_shard_payload(payload, expected_source):
    if not isinstance(payload, dict) or payload.get("schema_version") != SHARD_PAYLOAD_SCHEMA_VERSION:
        return None
    if payload.get("source") != expected_source:
        return None
    events = payload.get("events")
    if not isinstance(events, list):
        return None
    return len(events)


def _index_payload(shard_entries, skipped_sources, total_events):
    return {
        "cache_schema_version": SHARD_CACHE_SCHEMA_VERSION,
        "num_events": total_events,
        "shards": shard_entries,
        "skipped_sources": skipped_sources,
    }


def _resume_prefix_from_index(index_path, root_sources, resume_from_root_index):
    """Trust already indexed sources before a requested 1-based resume position."""
    if resume_from_root_index <= 1:
        return [], [], 0
    if not index_path.exists():
        raise ValueError(
            f"--resume-from-root-index={resume_from_root_index} requires an existing index: {index_path}"
        )

    prior_index = _load_json(index_path)
    prefix_sources = root_sources[: resume_from_root_index - 1]
    prefix_paths = {source["path"] for source in prefix_sources}
    shard_entries = [
        entry for entry in prior_index.get("shards", [])
        if entry.get("source", {}).get("path") in prefix_paths
    ]
    skipped_sources = [
        entry for entry in prior_index.get("skipped_sources", [])
        if entry.get("source", {}).get("path") in prefix_paths
    ]
    indexed_by_path = {entry.get("source", {}).get("path"): entry.get("source") for entry in shard_entries}
    skipped_by_path = {entry.get("source", {}).get("path"): entry.get("source") for entry in skipped_sources}
    accounted_sources = [
        indexed_by_path.get(source["path"], skipped_by_path.get(source["path"]))
        for source in prefix_sources
    ]
    if accounted_sources != prefix_sources:
        raise ValueError(
            f"Cannot resume at ROOT index {resume_from_root_index}: existing index does not account "
            "for every earlier ROOT source."
        )

    total_events = 0
    for entry in shard_entries:
        if entry.get("event_start") != total_events:
            raise ValueError(f"Cannot resume from non-contiguous shard event offsets in {index_path}")
        total_events = entry.get("event_stop")
    return shard_entries, skipped_sources, int(total_events)


def prepare_sharded_tensor_cache(
    cache_dir,
    root_specs,
    valid_labels,
    filter_noise=True,
    supervise_noise=False,
    force=False,
    skip_existing=True,
    max_root_files=None,
    max_events_per_root_file=None,
    read_step_size=500,
    skip_failed_root_files=False,
    resume_from_root_index=1,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
    logger=None,
):
    """Create or resume a one-ROOT-file-per-shard canonical tensor cache."""
    from ml_ldmx.datasets.ecal_tpad_loading import (
        attach_root_source_metadata,
        ecal_tpad_event_to_tensors,
    )

    logger = logger or logging.getLogger(__name__)
    if resume_from_root_index < 1:
        raise ValueError("resume_from_root_index must be at least 1.")
    cache_dir = Path(cache_dir)
    shards_dir = cache_dir / "shards"
    root_sources = _root_sources(root_specs, max_root_files=max_root_files)
    spec = _cache_spec(
        root_sources=root_sources,
        valid_labels=tuple(valid_labels),
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        max_events_per_root_file=max_events_per_root_file,
        ecal_energy_transform=ecal_energy_transform,
        tpad_pe_transform=tpad_pe_transform,
    )
    manifest_path = cache_dir / "manifest.json"
    index_path = cache_dir / "index.json"

    if manifest_path.exists() and not force:
        manifest = _load_json(manifest_path)
        if not _cache_specs_match(manifest.get("cache_spec"), spec):
            raise ValueError(
                f"Existing sharded cache metadata does not match requested ROOT inputs/settings: {cache_dir}. "
                "Choose a different --processed-cache or pass --force-sharded-cache."
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cache_schema_version": SHARD_CACHE_SCHEMA_VERSION,
        "format": "ecal_tpad_root_file_shards",
        "cache_spec": spec,
        "feature_layout": FEATURE_LAYOUT,
        "filter_noise": bool(filter_noise),
        "supervise_noise": bool(supervise_noise),
        "ecal_energy_transform": ecal_energy_transform,
        "tpad_pe_transform": tpad_pe_transform,
        "valid_labels": list(valid_labels),
    }
    _write_json(manifest_path, manifest)

    prior_skipped_by_path = {}
    if skip_failed_root_files and index_path.exists() and not force:
        try:
            prior_index = _load_json(index_path)
            prior_skipped_by_path = {
                entry["source"]["path"]: entry
                for entry in prior_index.get("skipped_sources", [])
                if isinstance(entry, dict) and isinstance(entry.get("source"), dict)
            }
        except Exception:
            prior_skipped_by_path = {}

    shard_entries, skipped_sources, total_events = _resume_prefix_from_index(
        index_path,
        root_sources,
        resume_from_root_index,
    )
    if resume_from_root_index > 1:
        logger.info(
            "Fast resume: trusting %s indexed shard(s) and %s recorded skipped ROOT file(s) "
            "before ROOT index %s without loading shard payloads.",
            len(shard_entries),
            len(skipped_sources),
            resume_from_root_index,
        )
    for shard_idx, source in enumerate(
        root_sources[resume_from_root_index - 1 :],
        start=resume_from_root_index,
    ):
        shard_path = shards_dir / f"shard_{shard_idx:06d}.pt"
        num_events = None
        if skip_existing and shard_path.exists() and not force:
            try:
                num_events = _validate_shard_payload(_load_torch(shard_path), source)
            except Exception:
                num_events = None
            if num_events is not None:
                logger.info("Reusing valid processed shard: %s", shard_path.name)

        if num_events is None:
            prior_skip = prior_skipped_by_path.get(source["path"])
            if prior_skip is not None and prior_skip.get("source") == source:
                skipped_sources.append(prior_skip)
                logger.warning(
                    "Reusing previously recorded skipped ROOT file: %s (%s: %s)",
                    source["name"],
                    prior_skip.get("error_type", "unknown error"),
                    prior_skip.get("error", "no message"),
                )
                _write_json(index_path, _index_payload(shard_entries, skipped_sources, total_events))
                continue

            logger.info("Tensorizing ROOT file into shard: %s -> %s", source["name"], shard_path.name)
            events = []
            try:
                for local_entry, raw_event in iter_ecal_rechits_with_truth_and_triggerpad_context(
                    source["path"],
                    max_events=max_events_per_root_file,
                    step_size=read_step_size,
                ):
                    event = ecal_tpad_event_to_tensors(
                        raw_event,
                        event_idx=total_events + len(events),
                        valid_labels=tuple(valid_labels),
                        target_mode="physical-origin",
                        filter_noise=filter_noise,
                        supervise_noise=supervise_noise,
                        ecal_energy_transform=ecal_energy_transform,
                        tpad_pe_transform=tpad_pe_transform,
                    )
                    attach_root_source_metadata(
                        event,
                        {"file": source["name"], "entry": local_entry},
                        global_event_idx=total_events + len(events),
                        electron_count=source["electron_count"],
                        source_label=source["source_label"],
                    )
                    events.append(event)
            except Exception as exc:
                if not skip_failed_root_files:
                    raise
                skipped_source = {
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                skipped_sources.append(skipped_source)
                logger.exception("Skipping failed ROOT file and continuing: %s", source["path"])
                _write_json(index_path, _index_payload(shard_entries, skipped_sources, total_events))
                continue
            payload = {
                "schema_version": SHARD_PAYLOAD_SCHEMA_VERSION,
                "source": source,
                "events": events,
            }
            torch.save(payload, shard_path)
            num_events = len(events)
            logger.info("Wrote %s event(s) to %s", num_events, shard_path)

        entry = {
            "path": str(shard_path.relative_to(cache_dir)).replace("\\", "/"),
            "source": source,
            "num_events": int(num_events),
            "event_start": total_events,
            "event_stop": total_events + int(num_events),
        }
        shard_entries.append(entry)
        total_events += int(num_events)
        _write_json(index_path, _index_payload(shard_entries, skipped_sources, total_events))

    if not shard_entries:
        raise ValueError("No ROOT files were selected for sharded preprocessing.")
    logger.info(
        "Sharded tensor cache ready: %s event(s) in %s shard(s); skipped ROOT files: %s",
        total_events,
        len(shard_entries),
        len(skipped_sources),
    )
    return cache_dir


def validate_sharded_tensor_cache(cache_dir, load_shards=True, allow_incomplete=False):
    """Validate manifest/index presence and optionally load every listed shard."""
    cache_dir = Path(cache_dir)
    if not has_sharded_tensor_cache(cache_dir):
        raise ValueError(f"Sharded cache requires manifest.json and index.json: {cache_dir}")
    manifest = _load_json(cache_dir / "manifest.json")
    index = _load_json(cache_dir / "index.json")
    entries = index.get("shards", [])
    if not entries:
        raise ValueError(f"Sharded cache index contains no shards: {cache_dir}")
    expected_sources = manifest.get("cache_spec", {}).get("root_sources", [])
    skipped_sources = index.get("skipped_sources", [])
    indexed_sources = [entry.get("source") for entry in entries]
    skipped_input_sources = [entry.get("source") for entry in skipped_sources]
    accounted_sources = []
    indexed_idx = 0
    skipped_idx = 0
    for expected_source in expected_sources:
        if indexed_idx < len(indexed_sources) and indexed_sources[indexed_idx] == expected_source:
            accounted_sources.append(expected_source)
            indexed_idx += 1
        elif skipped_idx < len(skipped_input_sources) and skipped_input_sources[skipped_idx] == expected_source:
            accounted_sources.append(expected_source)
            skipped_idx += 1
        else:
            break
    all_index_entries_accounted = indexed_idx == len(indexed_sources) and skipped_idx == len(skipped_input_sources)
    if allow_incomplete:
        sources_match = all_index_entries_accounted
    else:
        sources_match = all_index_entries_accounted and accounted_sources == expected_sources
    if not sources_match:
        raise ValueError(f"Sharded cache index is incomplete or does not match its manifest: {cache_dir}")

    total_events = 0
    for entry in entries:
        shard_path = cache_dir / entry["path"]
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing processed shard listed in index: {shard_path}")
        if entry["event_start"] != total_events:
            raise ValueError(f"Non-contiguous shard event offsets in {cache_dir / 'index.json'}")
        if load_shards:
            payload = _load_torch(shard_path)
            num_events = _validate_shard_payload(payload, entry["source"])
            if num_events is None or num_events != entry["num_events"]:
                raise ValueError(f"Invalid processed shard payload: {shard_path}")
        total_events = entry["event_stop"]
    if total_events != index.get("num_events"):
        raise ValueError(f"Sharded cache event count does not match index: {cache_dir}")
    return manifest, index


def create_parallel_shard_plan(
    plan_path,
    output_root,
    root_specs,
    valid_labels=(1, 2, 3),
    filter_noise=False,
    supervise_noise=True,
    max_root_files=None,
    max_events_per_root_file=None,
    ecal_energy_transform="log1p",
    tpad_pe_transform="log1p",
):
    """Freeze deterministic, parallel-safe work for one or more ROOT sources."""
    output_root = Path(output_root).resolve()
    plan_path = Path(plan_path).resolve()
    status_dir = plan_path.parent / f"{plan_path.stem}_status"
    tasks = []
    source_groups = []

    for electron_count, source_label, source_dir in root_specs:
        sources = _root_sources(
            [(electron_count, source_label, source_dir)],
            max_root_files=max_root_files,
        )
        cache_dir = output_root / source_label / "events"
        source_groups.append(
            {
                "electron_count": int(electron_count) if electron_count is not None else None,
                "source_label": source_label,
                "cache_dir": str(cache_dir),
                "root_sources": sources,
            }
        )
        for class_shard_index, source in enumerate(sources, start=1):
            task_index = len(tasks)
            tasks.append(
                {
                    "task_index": task_index,
                    "class_shard_index": class_shard_index,
                    "electron_count": source["electron_count"],
                    "source_label": source_label,
                    "source": source,
                    "cache_dir": str(cache_dir),
                    "shard_path": str(cache_dir / "shards" / f"shard_{class_shard_index:06d}.pt"),
                    "status_path": str(status_dir / f"task_{task_index:06d}.json"),
                }
            )

    if not tasks:
        raise ValueError("No ROOT files were selected for parallel preprocessing.")

    preprocessing_spec = _cache_spec(
        root_sources=[],
        valid_labels=tuple(valid_labels),
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        max_events_per_root_file=max_events_per_root_file,
        ecal_energy_transform=ecal_energy_transform,
        tpad_pe_transform=tpad_pe_transform,
    )
    preprocessing_spec.pop("root_sources")
    plan = {
        "plan_schema_version": PARALLEL_PLAN_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "status_dir": str(status_dir),
        "preprocessing_spec": preprocessing_spec,
        "source_groups": source_groups,
        "tasks": tasks,
    }
    status_dir.mkdir(parents=True, exist_ok=True)
    _write_json(plan_path, plan)
    return plan


def load_parallel_shard_plan(plan_path):
    plan_path = Path(plan_path)
    plan = _load_json(plan_path)
    if plan.get("plan_schema_version") != PARALLEL_PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported parallel shard plan: {plan_path}")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Parallel shard plan contains no tasks: {plan_path}")
    if [task.get("task_index") for task in tasks] != list(range(len(tasks))):
        raise ValueError(f"Parallel shard task indices are not contiguous: {plan_path}")
    return plan


def _parallel_payload_is_valid(payload, task, preprocessing_spec):
    num_events = _validate_shard_payload(payload, task["source"])
    if (
        num_events is None
        or num_events == 0
        or payload.get("preprocessing_spec") != preprocessing_spec
    ):
        return None
    return num_events


def prepare_parallel_shard_task(
    plan_path,
    task_index,
    read_step_size=500,
    force=False,
    logger=None,
):
    """Tensorize one plan task without writing shared cache metadata."""
    from ml_ldmx.datasets.ecal_tpad_loading import (
        attach_root_source_metadata,
        ecal_tpad_event_to_tensors,
    )

    logger = logger or logging.getLogger(__name__)
    plan = load_parallel_shard_plan(plan_path)
    task_index = int(task_index)
    if task_index < 0 or task_index >= len(plan["tasks"]):
        raise IndexError(f"Task index {task_index} is outside 0..{len(plan['tasks']) - 1}.")
    task = plan["tasks"][task_index]
    spec = plan["preprocessing_spec"]
    source = task["source"]
    shard_path = Path(task["shard_path"])
    status_path = Path(task["status_path"])
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        num_events = None
        reused = False
        if shard_path.exists() and not force:
            try:
                num_events = _parallel_payload_is_valid(_load_torch(shard_path), task, spec)
            except Exception:
                num_events = None
            reused = num_events is not None

        if num_events is None:
            logger.info("Tensorizing %s -> %s", source["path"], shard_path)
            events = []
            for local_entry, raw_event in iter_ecal_rechits_with_truth_and_triggerpad_context(
                source["path"],
                max_events=spec["max_events_per_root_file"],
                step_size=read_step_size,
            ):
                local_event_idx = len(events)
                event = ecal_tpad_event_to_tensors(
                    raw_event,
                    event_idx=local_event_idx,
                    valid_labels=tuple(spec["valid_labels"]),
                    target_mode="physical-origin",
                    filter_noise=spec["filter_noise"],
                    supervise_noise=spec["supervise_noise"],
                    ecal_energy_transform=spec["ecal_energy_transform"],
                    tpad_pe_transform=spec["tpad_pe_transform"],
                )
                attach_root_source_metadata(
                    event,
                    {"file": source["name"], "entry": local_entry},
                    global_event_idx=local_event_idx,
                    electron_count=source["electron_count"],
                    source_label=source["source_label"],
                )
                events.append(event)

            payload = {
                "schema_version": SHARD_PAYLOAD_SCHEMA_VERSION,
                "source": source,
                "preprocessing_spec": spec,
                "events": events,
            }
            num_events = _parallel_payload_is_valid(payload, task, spec)
            if num_events is None:
                raise ValueError(f"Worker produced an invalid in-memory shard for {source['path']}")
            temporary_path = shard_path.with_suffix(shard_path.suffix + f".task-{task_index}.tmp")
            torch.save(payload, temporary_path)
            temporary_path.replace(shard_path)
            if not shard_path.exists() or shard_path.stat().st_size == 0:
                raise ValueError(f"Worker did not write a non-empty shard: {shard_path}")

        status = {
            "status": "complete",
            "task_index": task_index,
            "source": source,
            "shard_path": str(shard_path),
            "num_events": int(num_events),
            "reused": reused,
        }
        _write_json(status_path, status)
        logger.info(
            "%s task %s with %s event(s): %s",
            "Reused" if reused else "Completed",
            task_index,
            num_events,
            shard_path,
        )
        return status
    except Exception as exc:
        _write_json(
            status_path,
            {
                "status": "failed",
                "task_index": task_index,
                "source": source,
                "shard_path": str(shard_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def _validate_ml_ready_event(event, ecal_energy_transform, tpad_pe_transform):
    from ml_ldmx.datasets.model_views import validate_canonical_combined_event
    from ml_ldmx.datasets.tensorize import transform_ecal_energy, transform_tpad_pe

    validate_canonical_combined_event(event)
    for key in ("ecal_raw_energy", "tpad_raw_pe"):
        if key not in event:
            raise ValueError(f"ML-ready shard event is missing preserved field {key!r}.")

    ecal_raw_energy = event["ecal_raw_energy"].to(dtype=torch.float32)
    expected_ecal = transform_ecal_energy(ecal_raw_energy, mode=ecal_energy_transform)
    if not torch.allclose(event["ecal_input_energy"], expected_ecal):
        raise ValueError("Stored ECal input energy does not match its declared transform.")
    if not torch.allclose(event["x"][event["ecal_mask"], 5], expected_ecal):
        raise ValueError("Combined ECal feature column does not match preserved raw energy.")

    tpad_raw_pe = event["tpad_raw_pe"].to(dtype=torch.float32)
    expected_tpad = transform_tpad_pe(tpad_raw_pe, mode=tpad_pe_transform)
    if not torch.allclose(event["tpad"][:, 1], expected_tpad):
        raise ValueError("Stored TriggerPad pe input does not match its declared transform.")
    if not torch.allclose(event["x"][event["tpad_mask"], 7], expected_tpad):
        raise ValueError("Combined TriggerPad feature column does not match preserved raw pe.")


def validate_ml_ready_sharded_cache(cache_dir, load_all_shards=False):
    """Validate cache structure and sampled raw-to-input feature transforms."""
    manifest, index = validate_sharded_tensor_cache(cache_dir, load_shards=load_all_shards)
    spec = manifest["cache_spec"]
    dataset = ShardedECalTpadDataset(cache_dir, shard_cache_size=1)
    if len(dataset) == 0:
        raise ValueError(f"ML-ready shard cache contains no events: {cache_dir}")
    sample_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for event_index in sample_indices:
        _validate_ml_ready_event(
            dataset[event_index],
            ecal_energy_transform=spec["ecal_energy_transform"],
            tpad_pe_transform=spec["tpad_pe_transform"],
        )
    return manifest, index


def finalize_parallel_shard_plan(
    plan_path,
    expected_events_by_label=None,
    allow_failed_root_files=False,
    load_all_shards=False,
):
    """Assemble cache metadata only after all planned workers are accounted for."""
    plan = load_parallel_shard_plan(plan_path)
    expected_events_by_label = expected_events_by_label or {}
    tasks_by_label = {}
    problems = []

    for task in plan["tasks"]:
        tasks_by_label.setdefault(task["source_label"], []).append(task)
        status_path = Path(task["status_path"])
        if not status_path.exists():
            problems.append(f"task {task['task_index']}: missing status for {task['source']['path']}")
            continue
        status = _load_json(status_path)
        if status.get("task_index") != task["task_index"] or status.get("source") != task["source"]:
            problems.append(f"task {task['task_index']}: status does not match the frozen plan")
        elif status.get("status") != "complete" and not allow_failed_root_files:
            problems.append(
                f"task {task['task_index']}: {status.get('error_type', 'failed')}: "
                f"{status.get('error', 'no error message')}"
            )

    if problems:
        preview = "\n".join(f"  - {problem}" for problem in problems[:20])
        remainder = "" if len(problems) <= 20 else f"\n  - ... and {len(problems) - 20} more"
        raise RuntimeError(f"Cannot finalize incomplete parallel preprocessing:\n{preview}{remainder}")

    summaries = []
    for group in plan["source_groups"]:
        source_label = group["source_label"]
        cache_dir = Path(group["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        shard_entries = []
        skipped_sources = []
        total_events = 0

        tasks = sorted(tasks_by_label[source_label], key=lambda task: task["class_shard_index"])
        for task in tasks:
            status = _load_json(task["status_path"])
            if status.get("status") != "complete":
                skipped_sources.append(
                    {
                        "source": task["source"],
                        "error_type": status.get("error_type", "WorkerError"),
                        "error": status.get("error", "Worker did not complete."),
                    }
                )
                continue

            shard_path = Path(task["shard_path"])
            if not shard_path.exists() or shard_path.stat().st_size == 0:
                raise ValueError(f"Completed worker shard is missing or empty: {shard_path}")
            num_events = status.get("num_events")
            if not isinstance(num_events, int) or num_events < 0:
                raise ValueError(f"Completed worker has an invalid event count: {task['status_path']}")
            shard_entries.append(
                {
                    "path": str(shard_path.relative_to(cache_dir)).replace("\\", "/"),
                    "source": task["source"],
                    "num_events": int(num_events),
                    "event_start": total_events,
                    "event_stop": total_events + int(num_events),
                }
            )
            total_events += int(num_events)

        expected_events = expected_events_by_label.get(source_label)
        if expected_events not in (None, 0) and total_events != int(expected_events):
            raise ValueError(
                f"{source_label} produced {total_events} events; expected {int(expected_events)}. "
                "The cache was not finalized."
            )
        if not shard_entries:
            raise ValueError(f"No successful shards are available for {source_label}.")

        root_sources = group["root_sources"]
        spec = dict(plan["preprocessing_spec"])
        spec["root_sources"] = root_sources
        manifest = {
            "cache_schema_version": SHARD_CACHE_SCHEMA_VERSION,
            "format": "ecal_tpad_root_file_shards",
            "cache_spec": spec,
            "feature_layout": FEATURE_LAYOUT,
            "filter_noise": spec["filter_noise"],
            "supervise_noise": spec["supervise_noise"],
            "ecal_energy_transform": spec["ecal_energy_transform"],
            "tpad_pe_transform": spec["tpad_pe_transform"],
            "valid_labels": spec["valid_labels"],
            "parallel_plan": str(Path(plan_path).resolve()),
        }
        _write_json(cache_dir / "manifest.json", manifest)
        _write_json(
            cache_dir / "index.json",
            _index_payload(shard_entries, skipped_sources, total_events),
        )
        validate_ml_ready_sharded_cache(cache_dir, load_all_shards=load_all_shards)
        summaries.append(
            {
                "electron_count": group["electron_count"],
                "source_label": source_label,
                "cache_dir": str(cache_dir),
                "num_root_files": len(root_sources),
                "num_shards": len(shard_entries),
                "num_events": total_events,
                "num_skipped_root_files": len(skipped_sources),
            }
        )

    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(Path(plan_path).resolve()),
        "preprocessing_spec": plan["preprocessing_spec"],
        "sources": summaries,
    }
    _write_json(Path(plan["output_root"]) / "preprocessing_summary.json", summary)
    return summary


class ShardedECalTpadDataset(Dataset):
    """Lazy event dataset backed by one tensor shard per source ROOT file."""

    def __init__(
        self,
        cache_dir,
        max_events=None,
        shard_cache_size=1,
        event_transform=None,
        allow_incomplete=False,
    ):
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive.")
        self.cache_dir = Path(cache_dir)
        self.metadata, self.index = validate_sharded_tensor_cache(
            self.cache_dir,
            load_shards=False,
            allow_incomplete=allow_incomplete,
        )
        self.shards = self.index["shards"]
        self.event_stops = [entry["event_stop"] for entry in self.shards]
        available_events = int(self.index["num_events"])
        self.num_events = available_events if max_events is None else min(int(max_events), available_events)
        self.shard_cache_size = int(shard_cache_size)
        self.event_transform = event_transform
        self._loaded_shards = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def __len__(self):
        return self.num_events

    @property
    def root_files(self):
        return [Path(entry["source"]["path"]) for entry in self.shards]

    @property
    def source_files(self):
        return [entry["source"]["name"] for entry in self.shards]

    @property
    def shard_event_counts(self):
        """Event counts for shards that are reachable under this dataset's max-event limit."""
        remaining = self.num_events
        counts = []
        for entry in self.shards:
            if remaining <= 0:
                break
            count = min(int(entry["num_events"]), remaining)
            counts.append(count)
            remaining -= count
        return counts

    def set_event_transform(self, event_transform):
        self.event_transform = event_transform

    def set_shard_cache_size(self, shard_cache_size):
        """Resize the in-memory shard LRU without changing dataset contents."""
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive.")
        self.shard_cache_size = int(shard_cache_size)
        while len(self._loaded_shards) > self.shard_cache_size:
            self._loaded_shards.popitem(last=False)
            self._cache_evictions += 1

    def clear_cache(self):
        """Drop loaded shard payloads; useful before memory-sensitive phases."""
        self._loaded_shards.clear()

    def cache_info(self):
        """Return lightweight shard-LRU telemetry for logs and run overviews."""
        return {
            "kind": "sharded",
            "cache_dir": str(self.cache_dir),
            "num_events": int(self.num_events),
            "num_shards": len(self.shards),
            "shard_cache_size": int(self.shard_cache_size),
            "loaded_shards": list(self._loaded_shards.keys()),
            "loaded_shard_paths": [
                self.shards[shard_idx]["path"] for shard_idx in self._loaded_shards
            ],
            "cache_hits": int(self._cache_hits),
            "cache_misses": int(self._cache_misses),
            "cache_evictions": int(self._cache_evictions),
            "shard_event_counts": self.shard_event_counts,
        }

    def _shard_idx_for_event(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect_right(self.event_stops, index)

    def _load_shard(self, shard_idx):
        if shard_idx in self._loaded_shards:
            self._cache_hits += 1
            payload = self._loaded_shards.pop(shard_idx)
            self._loaded_shards[shard_idx] = payload
            return payload
        self._cache_misses += 1
        payload = _load_torch(self.cache_dir / self.shards[shard_idx]["path"])
        self._loaded_shards[shard_idx] = payload
        while len(self._loaded_shards) > self.shard_cache_size:
            self._loaded_shards.popitem(last=False)
            self._cache_evictions += 1
        return payload

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        shard_idx = self._shard_idx_for_event(index)
        entry = self.shards[shard_idx]
        local_index = index - entry["event_start"]
        event = dict(self._load_shard(shard_idx)["events"][local_index])
        event["event_idx"] = torch.tensor(index, dtype=torch.long)
        if self.event_transform is not None:
            event = self.event_transform(event)
        return event

    def order_indices_for_access(self, indices, seed=None):
        """Group accesses by shard to avoid repeatedly opening large shard files."""
        grouped = {}
        for index in indices:
            grouped.setdefault(self._shard_idx_for_event(int(index)), []).append(int(index))
        shard_indices = list(grouped)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
            order = torch.randperm(len(shard_indices), generator=generator).tolist()
            shard_indices = [shard_indices[idx] for idx in order]
        else:
            shard_indices.sort()
        ordered = []
        for shard_idx in shard_indices:
            group = grouped[shard_idx]
            if seed is not None and len(group) > 1:
                order = torch.randperm(len(group), generator=generator).tolist()
                group = [group[idx] for idx in order]
            ordered.extend(group)
        return ordered


class MultiShardedECalTpadDataset(Dataset):
    """Lazy view over multiple independent sharded caches."""

    def __init__(self, sources, max_events=None, event_transform=None):
        if not sources:
            raise ValueError("MultiShardedECalTpadDataset requires at least one source.")
        self.sources = list(sources)
        self.event_transform = event_transform
        self.offsets = []
        total = 0
        for source in self.sources:
            self.offsets.append(total)
            total += len(source["dataset"])
        self.num_events = total if max_events is None else min(int(max_events), total)

    def __len__(self):
        return self.num_events

    @property
    def root_files(self):
        files = []
        for source in self.sources:
            files.extend(source["dataset"].root_files)
        return files

    @property
    def source_files(self):
        files = []
        for source in self.sources:
            files.extend(source["dataset"].source_files)
        return files

    @property
    def source_summaries(self):
        return [
            {
                "electron_count": source["electron_count"],
                "source_label": source["source_label"],
                "cache_dir": str(source["cache_dir"]),
                "num_events": len(source["dataset"]),
            }
            for source in self.sources
        ]

    @property
    def shard_event_counts(self):
        counts = []
        for source in self.sources:
            counts.extend(source["dataset"].shard_event_counts)
        return counts

    def set_event_transform(self, event_transform):
        self.event_transform = event_transform

    def set_shard_cache_size(self, shard_cache_size):
        """Resize each source dataset's shard LRU."""
        for source in self.sources:
            source["dataset"].set_shard_cache_size(shard_cache_size)

    def clear_cache(self):
        """Drop loaded shard payloads from every source dataset."""
        for source in self.sources:
            source["dataset"].clear_cache()

    def cache_info(self):
        """Return cache telemetry for the combined lazy sharded dataset."""
        source_infos = []
        for source in self.sources:
            info = source["dataset"].cache_info()
            info.update(
                {
                    "electron_count": source["electron_count"],
                    "source_label": source["source_label"],
                    "source_cache_dir": str(source["cache_dir"]),
                }
            )
            source_infos.append(info)
        return {
            "kind": "multi_sharded",
            "num_events": int(self.num_events),
            "num_sources": len(self.sources),
            "sources": source_infos,
            "cache_hits": sum(info["cache_hits"] for info in source_infos),
            "cache_misses": sum(info["cache_misses"] for info in source_infos),
            "cache_evictions": sum(info["cache_evictions"] for info in source_infos),
            "shard_event_counts": self.shard_event_counts,
        }

    def _source_idx_for_event(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect_right(self.offsets, index) - 1

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        source_idx = self._source_idx_for_event(index)
        source = self.sources[source_idx]
        local_index = index - self.offsets[source_idx]
        event = dict(source["dataset"][local_index])
        event["event_idx"] = torch.tensor(index, dtype=torch.long)
        if source["electron_count"] is not None:
            event["electron_count"] = torch.tensor(int(source["electron_count"]), dtype=torch.long)
        event["source_label"] = source["source_label"]
        if self.event_transform is not None:
            event = self.event_transform(event)
        return event

    def order_indices_for_access(self, indices, seed=None):
        grouped = {}
        for index in indices:
            source_idx = self._source_idx_for_event(int(index))
            grouped.setdefault(source_idx, []).append(int(index))
        source_indices = list(grouped)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
            order = torch.randperm(len(source_indices), generator=generator).tolist()
            source_indices = [source_indices[idx] for idx in order]
        else:
            source_indices.sort()

        ordered = []
        for source_idx in source_indices:
            group = grouped[source_idx]
            local_group = [index - self.offsets[source_idx] for index in group]
            local_order = self.sources[source_idx]["dataset"].order_indices_for_access(
                local_group,
                seed=None if seed is None else int(seed) + source_idx + 1,
            )
            ordered.extend(self.offsets[source_idx] + local_index for local_index in local_order)
        return ordered
