"""Target preparation and padded batching for the ECal/TPad slot model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


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

    if target.numel() == 0:
        raise ValueError("An ECal/TPad slot-model event must contain at least one ECal target.")
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
        target = F.one_hot(
            origin_target.clamp(0, max_electrons),
            num_classes=num_classes,
        ).float()
    else:
        fraction_target = event["fraction_target"].to(dtype=torch.float32)
        if fraction_target.ndim != 2:
            raise ValueError(
                "Expected event['fraction_target'] with shape [num_ecal, num_classes]."
            )
        if fraction_target.shape[1] == num_classes:
            target = fraction_target
        elif fraction_target.shape[1] == max_electrons:
            noise_column = torch.zeros(
                (fraction_target.shape[0], 1),
                dtype=fraction_target.dtype,
                device=fraction_target.device,
            )
            target = torch.cat([noise_column, fraction_target], dim=1)
        else:
            raise ValueError(
                f"Expected fraction_target with {max_electrons} or {num_classes} columns, "
                f"got {fraction_target.shape[1]}."
            )

    if target.shape[0] != origin_target.shape[0]:
        raise ValueError(
            "Origin and fraction targets must describe the same number of ECal hits: "
            f"{origin_target.shape[0]} vs {target.shape[0]}."
        )
    noise_mask = event.get("is_noise_target")
    if noise_mask is not None:
        noise_mask = noise_mask.to(dtype=torch.bool)
        if noise_mask.shape != (target.shape[0],):
            raise ValueError("event['is_noise_target'] must align with fraction targets.")
        target = target.clone()
        target[noise_mask] = 0.0
        target[noise_mask, 0] = 1.0
    return target


def slot_targets_from_event(
    event: dict,
    origin_target: torch.Tensor,
    fraction_target: torch.Tensor,
    max_electrons: int,
) -> torch.Tensor:
    valid = torch.zeros((max_electrons,), dtype=torch.float32, device=origin_target.device)
    for slot_idx in range(max_electrons):
        class_idx = slot_idx + 1
        # A slot is valid if it owns any hard-label hit or any soft target mass.
        has_hard_hit = bool((origin_target == class_idx).any().item())
        has_fraction_mass = bool((fraction_target[:, class_idx].sum() > 0.0).item())
        valid[slot_idx] = 1.0 if has_hard_hit or has_fraction_mass else 0.0
    return valid


def count_target_from_event(
    event: dict,
    slot_target: torch.Tensor,
    max_electrons: int,
) -> torch.Tensor:
    for key in ("electron_count", "event_electron_count", "count_target", "num_electrons"):
        if key in event:
            value = event[key]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().reshape(-1)[0].item())
            else:
                value = int(value)
            return torch.tensor(min(max(value, 0), max_electrons), dtype=torch.long)
    return slot_target.sum().detach().cpu().to(dtype=torch.long).clamp(max=max_electrons)


@dataclass
class ECalTpadSlotBatch:
    """Padded inputs and aligned token/event supervision for one slot-model batch."""

    x: torch.Tensor
    valid_mask: torch.Tensor
    ecal_mask: torch.Tensor
    origin_target: torch.Tensor
    fraction_target: torch.Tensor
    slot_target: torch.Tensor
    count_target: torch.Tensor
    events: list[dict]

    @property
    def batch_size(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_hits(self) -> int:
        return int(self.ecal_mask.sum().item())

    def to(self, device):
        return ECalTpadSlotBatch(
            x=self.x.to(device=device, dtype=torch.float32),
            valid_mask=self.valid_mask.to(device=device, dtype=torch.bool),
            ecal_mask=self.ecal_mask.to(device=device, dtype=torch.bool),
            origin_target=self.origin_target.to(device=device, dtype=torch.long),
            fraction_target=self.fraction_target.to(device=device, dtype=torch.float32),
            slot_target=self.slot_target.to(device=device, dtype=torch.float32),
            count_target=self.count_target.to(device=device, dtype=torch.long),
            events=self.events,
        )


def collate_ecal_tpad_slot_batch(
    events: list[dict],
    max_electrons: int,
    ignore_index: int = IGNORE_INDEX,
) -> ECalTpadSlotBatch:
    """Pad variable-length events without turning padding into model input or supervision."""
    if not events:
        raise ValueError("Cannot collate an empty ECal/TPad slot-model batch.")

    first_x = events[0]["x"]
    if first_x.ndim != 2 or first_x.shape[0] == 0:
        raise ValueError(
            f"Expected event['x'] with non-empty shape [N, F], got {tuple(first_x.shape)}."
        )
    feature_dim = int(first_x.shape[1])
    max_tokens = max(int(event["x"].shape[0]) for event in events)
    batch_size = len(events)
    num_classes = max_electrons + 1

    x = torch.zeros((batch_size, max_tokens, feature_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    ecal_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    origin_target = torch.full(
        (batch_size, max_tokens),
        int(ignore_index),
        dtype=torch.long,
    )
    fraction_target = torch.zeros(
        (batch_size, max_tokens, num_classes),
        dtype=torch.float32,
    )
    slot_target = torch.zeros((batch_size, max_electrons), dtype=torch.float32)
    count_target = torch.zeros((batch_size,), dtype=torch.long)

    for row, event in enumerate(events):
        event_x = event["x"]
        if event_x.ndim != 2 or event_x.shape[0] == 0:
            raise ValueError(
                f"Expected event['x'] with non-empty shape [N, F], got {tuple(event_x.shape)}."
            )
        if int(event_x.shape[1]) != feature_dim:
            raise ValueError(
                "All events in a slot-model batch must have the same feature dimension."
            )

        num_tokens = int(event_x.shape[0])
        event_ecal_mask = ecal_mask_from_event(event).detach().cpu()
        if event_ecal_mask.shape != (num_tokens,):
            raise ValueError(
                f"Expected ecal_mask with shape [{num_tokens}], got {tuple(event_ecal_mask.shape)}."
            )
        num_ecal = int(event_ecal_mask.sum().item())
        event_origin_target = origin_targets_from_event(event, max_electrons).detach().cpu()
        event_fraction_target = fraction_targets_from_event(
            event,
            event_origin_target,
            max_electrons,
        ).detach().cpu()
        if event_origin_target.shape != (num_ecal,):
            raise ValueError(
                "Origin targets must align with the ECal mask: "
                f"{event_origin_target.shape[0]} targets for {num_ecal} ECal tokens."
            )
        event_slot_target = slot_targets_from_event(
            event,
            event_origin_target,
            event_fraction_target,
            max_electrons,
        ).detach().cpu()
        event_count_target = count_target_from_event(
            event,
            event_slot_target,
            max_electrons,
        ).detach().cpu()

        x[row, :num_tokens] = event_x.detach().cpu().to(dtype=torch.float32)
        valid_mask[row, :num_tokens] = True
        ecal_mask[row, :num_tokens] = event_ecal_mask
        row_ecal_mask = ecal_mask[row]
        origin_target[row, row_ecal_mask] = event_origin_target
        fraction_target[row, row_ecal_mask] = event_fraction_target
        slot_target[row] = event_slot_target
        count_target[row] = event_count_target

    return ECalTpadSlotBatch(
        x=x,
        valid_mask=valid_mask,
        ecal_mask=ecal_mask,
        origin_target=origin_target,
        fraction_target=fraction_target,
        slot_target=slot_target,
        count_target=count_target,
        events=list(events),
    )
