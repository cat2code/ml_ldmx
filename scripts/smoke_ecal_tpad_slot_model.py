"""
Smoke test for the ECal + TriggerPadTracks slot-validity model.

Examples from the repository root:

    python mldmx/scripts/smoke_ecal_tpad_slot_model.py --max-events 3

Example from the mldmx directory:

    python scripts/smoke_ecal_tpad_slot_model.py --max-events 3
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_dataset import ECalTriggerPadTensorDataset
from mldmx.datasets.ecal_tpad_loading import load_ecal_tpad_tensor_events
from mldmx.io.root_files import find_root_files
from mldmx.models import ECalTpadSlotModel
from mldmx.train.losses import soft_label_cross_entropy


VALID_LABELS = (1, 2, 3)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ecal_tpad_3class_smoke"
DEFAULT_ROOT_DIRS = (
    PROJECT_ROOT / "data/ldmx_overlay_events_700k/2e/events",
    PROJECT_ROOT / "data/ldmx_overlay_events_700k/3e/events",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run forward/backward smoke test for ECalTpadSlotModel."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string. Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    candidate = PROJECT_ROOT.parent / path
    if candidate.exists():
        return candidate
    return path


def has_processed_events(processed_dir: Path) -> bool:
    return processed_dir.exists() and any(processed_dir.glob("event_*.pt"))


def load_processed_events(processed_dir: Path, max_events: int) -> list[dict]:
    dataset = ECalTriggerPadTensorDataset(processed_dir)
    return [dataset[idx] for idx in range(min(max_events, len(dataset)))]


def load_root_events(max_events: int) -> list[dict]:
    logger = logging.getLogger("slot-smoke")
    events = []
    remaining = max_events
    existing_dirs = [path for path in DEFAULT_ROOT_DIRS if path.exists()]
    if not existing_dirs:
        tried = ", ".join(str(path) for path in DEFAULT_ROOT_DIRS)
        raise FileNotFoundError(f"No processed events found and no ROOT dirs exist. Tried: {tried}")

    for dir_idx, data_dir in enumerate(existing_dirs):
        if remaining <= 0:
            break
        per_dir = remaining
        if len(existing_dirs) - dir_idx > 1:
            per_dir = max(1, remaining // (len(existing_dirs) - dir_idx))
        root_files = find_root_files(data_dir)
        loaded, _sources = load_ecal_tpad_tensor_events(
            root_files=root_files,
            max_events=per_dir,
            valid_labels=VALID_LABELS,
            target_mode="physical-origin",
            filter_noise=True,
            allow_fewer_events=True,
            data_dir=data_dir,
            logger=logger,
            read_step_size=50,
        )
        events.extend(loaded)
        remaining = max_events - len(events)

    if not events:
        raise RuntimeError("Could not load any tensor events from processed cache or ROOT files.")
    return events


def load_events(processed_dir: Path, max_events: int) -> list[dict]:
    processed_dir = resolve_path(processed_dir)
    if has_processed_events(processed_dir):
        print(f"Loading processed tensor events from: {processed_dir}")
        return load_processed_events(processed_dir, max_events)

    print("Processed tensor events not found; falling back to ROOT loading.")
    return load_root_events(max_events)


def ecal_mask_from_event(event: dict) -> torch.Tensor:
    if "ecal_mask" in event:
        return event["ecal_mask"].to(dtype=torch.bool)
    if "num_ecal" in event:
        num_ecal = int(event["num_ecal"])
    elif "y" in event:
        num_ecal = int(event["y"].shape[0])
    else:
        raise KeyError("Event has neither ecal_mask, num_ecal, nor y to identify ECal nodes.")
    mask = torch.zeros((event["x"].shape[0],), dtype=torch.bool)
    mask[:num_ecal] = True
    return mask


def origin_targets_from_event(event: dict, max_electrons: int) -> torch.Tensor:
    if "physical_y" in event:
        target = event["physical_y"].to(dtype=torch.long)
    elif "y" in event:
        target = event["y"].to(dtype=torch.long) + 1
    else:
        raise KeyError("Event is missing both physical_y and y origin targets.")

    if int(target.min().item()) < 0 or int(target.max().item()) > max_electrons:
        raise ValueError(
            f"Origin targets must be in 0..{max_electrons}, got "
            f"{int(target.min().item())}..{int(target.max().item())}."
        )
    return target


def fraction_targets_from_event(
    event: dict,
    origin_target: torch.Tensor,
    max_electrons: int,
) -> torch.Tensor:
    num_classes = max_electrons + 1
    if "fraction_target" not in event:
        return F.one_hot(origin_target.clamp(0, max_electrons), num_classes=num_classes).float()

    fraction_target = event["fraction_target"].to(dtype=torch.float32)
    if fraction_target.shape[1] == num_classes:
        return fraction_target
    if fraction_target.shape[1] == max_electrons:
        noise_column = torch.zeros(
            (fraction_target.shape[0], 1),
            dtype=fraction_target.dtype,
            device=fraction_target.device,
        )
        return torch.cat([noise_column, fraction_target], dim=1)
    raise ValueError(
        f"Expected fraction_target with {max_electrons} or {num_classes} columns, "
        f"got {fraction_target.shape[1]}."
    )


def slot_targets(
    origin_target: torch.Tensor,
    fraction_target: torch.Tensor,
    max_electrons: int,
) -> torch.Tensor:
    valid = torch.zeros((max_electrons,), dtype=torch.float32, device=origin_target.device)
    for slot_idx in range(max_electrons):
        class_idx = slot_idx + 1
        # A slot is valid if it owns any hard-label hit or has any soft fraction mass.
        has_hard_hit = bool((origin_target == class_idx).any().item())
        has_fraction_mass = bool((fraction_target[:, class_idx].sum() > 0.0).item())
        valid[slot_idx] = 1.0 if has_hard_hit or has_fraction_mass else 0.0
    return valid


def compute_event_losses(model, event: dict, device: torch.device) -> dict[str, torch.Tensor]:
    x = event["x"].to(device=device, dtype=torch.float32)
    ecal_mask = ecal_mask_from_event(event).to(device)
    outputs = model(x, ecal_mask=ecal_mask)

    origin_target = origin_targets_from_event(event, model.max_electrons).to(device)
    fraction_target = fraction_targets_from_event(
        event,
        origin_target.detach().cpu(),
        model.max_electrons,
    ).to(device)
    slot_target = slot_targets(origin_target, fraction_target, model.max_electrons)
    count_target = slot_target.sum().to(dtype=torch.long).clamp(max=model.max_electrons)

    ecal_origin_logits = outputs["origin_logits"][ecal_mask]
    ecal_fraction_logits = outputs["fraction_logits"][ecal_mask]
    origin_loss = F.cross_entropy(ecal_origin_logits, origin_target)
    fraction_loss = soft_label_cross_entropy(ecal_fraction_logits, fraction_target)
    slot_loss = F.binary_cross_entropy_with_logits(outputs["slot_valid_logits"], slot_target)
    count_loss = F.cross_entropy(
        outputs["count_logits"].unsqueeze(0),
        count_target.unsqueeze(0),
    )
    total_loss = origin_loss + fraction_loss + 0.2 * slot_loss + 0.2 * count_loss

    return {
        "total_loss": total_loss,
        "origin_loss": origin_loss,
        "fraction_loss": fraction_loss,
        "slot_loss": slot_loss,
        "count_loss": count_loss,
        "count_target": count_target,
        "count_pred": outputs["count_logits"].argmax(dim=-1),
        "outputs": outputs,
        "x_shape": torch.tensor(x.shape, device="cpu"),
    }


def main():
    args = parse_args()
    if args.max_events <= 0:
        raise ValueError("--max-events must be positive.")
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    events = load_events(args.processed_dir, args.max_events)
    if not events:
        raise RuntimeError("No events were loaded.")

    input_dim = int(events[0]["x"].shape[1])
    model = ECalTpadSlotModel(
        in_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=4,
        max_electrons=3,
        dropout=0.0,
        use_type_embedding=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = None
    records = []
    for event in events[: args.max_events]:
        losses = compute_event_losses(model, event, device)
        loss_sum = losses["total_loss"] if loss_sum is None else loss_sum + losses["total_loss"]
        records.append(losses)

    batch_loss = loss_sum / max(1, len(records))
    batch_loss.backward()
    optimizer.step()

    first = records[0]
    output_shapes = {
        key: tuple(value.shape)
        for key, value in first["outputs"].items()
        if isinstance(value, torch.Tensor)
    }
    print(f"events tested: {len(records)}")
    print(f"input shape: {tuple(first['x_shape'].tolist())}")
    print(f"output shapes: {output_shapes}")
    print(f"true electron count: {int(first['count_target'].item())}")
    print(f"predicted count argmax: {int(first['count_pred'].item())}")
    print(f"total_loss: {float(batch_loss.detach().cpu()):.6f}")
    print(f"origin_loss: {float(first['origin_loss'].detach().cpu()):.6f}")
    print(f"fraction_loss: {float(first['fraction_loss'].detach().cpu()):.6f}")
    print(f"slot_loss: {float(first['slot_loss'].detach().cpu()):.6f}")
    print(f"count_loss: {float(first['count_loss'].detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
