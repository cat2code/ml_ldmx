from datetime import datetime
from pathlib import Path


def resolve_run_dir(args):
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return args.output_dir / run_name


def resolve_data_dir(data_dir, project_root=None, default_relative="data/ldmx_overlay_events_700k/3e/events"):
    if data_dir.exists():
        return data_dir

    roots = []
    if project_root is not None:
        roots.extend([Path(project_root), Path(project_root).parent])
    roots.append(Path.cwd())

    candidates = [root / data_dir for root in roots]
    if project_root is not None:
        candidates.append(Path(project_root) / default_relative)

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = candidate.resolve() if candidate.exists() else candidate.absolute()
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find data directory '{data_dir}'. Tried: "
        + ", ".join(str(candidate) for candidate in unique_candidates)
    )
