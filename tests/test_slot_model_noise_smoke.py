"""Optional ROOT-backed smoke test for explicit slot-model noise supervision."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from ml_ldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    load_grouped_root_tensor_events,
)
from ml_ldmx.eval.ecal_tpad_slot_model import evaluate
from ml_ldmx.models import ECalTpadSlotModel
from ml_ldmx.train.ecal_tpad_slot_model import compute_event_losses


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = (1, 2, 3)


def _project_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _root_data_root() -> Path:
    env_root = os.environ.get("ML_LDMX_ROOT_DATA")
    candidates = [_project_path(env_root)] if env_root else [
        PROJECT_ROOT / "data/ldmx_overlay_events_700k",
        PROJECT_ROOT.parent / "mldmx/data/ldmx_overlay_events_700k",
    ]
    for data_root in candidates:
        if data_root.exists():
            return data_root
    searched = ", ".join(str(path) for path in candidates)
    raise unittest.SkipTest(f"No ROOT data directory found. Set ML_LDMX_ROOT_DATA. Searched: {searched}")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _canonical_order_from_non_noise_hits(event: dict) -> list[int]:
    noise_mask = event["is_noise_target"]
    origin_ids = event["origin_id_y"]
    positions = event["ecal_pos"]
    means = []
    for origin_id in sorted({int(value) for value in origin_ids[~noise_mask].tolist()}):
        mask = (~noise_mask) & (origin_ids == origin_id)
        means.append((origin_id, float(positions[mask, 1].mean().item())))
    return [origin_id for origin_id, _mean in sorted(means, key=lambda item: (item[1], item[0]))]


def _loss_args() -> SimpleNamespace:
    return SimpleNamespace(
        lambda_origin=1.0,
        lambda_fraction=1.0,
        lambda_slot=0.5,
        lambda_count=1.0,
        origin_class_weights=None,
        count_class_weights=None,
        batch_size=1,
        max_electrons=len(VALID_LABELS),
    )


class SlotModelNoiseSmokeTest(unittest.TestCase):
    def test_noise_targets_losses_and_evaluation_confusion(self):
        torch.manual_seed(7)
        data_root = _root_data_root()
        electron_count = _env_int("ML_LDMX_NOISE_ELECTRON_COUNT", 3)
        events_to_scan = _env_int("ML_LDMX_NOISE_EVENTS_TO_SCAN", 10)
        source_dir = data_root / f"{electron_count}e/events"
        if not source_dir.exists():
            raise unittest.SkipTest(f"Missing ROOT source directory: {source_dir}")

        events, _sources, _files = load_grouped_root_tensor_events(
            root_specs=[(electron_count, f"{electron_count}e", source_dir)],
            events_per_source=events_to_scan,
            valid_labels=VALID_LABELS,
            filter_noise=False,
            supervise_noise=True,
            allow_fewer_events=True,
            disable_progress=True,
            read_step_size=50,
        )
        noisy_events = [event for event in events if bool(event["is_noise_target"].any().item())]
        if not noisy_events:
            raise AssertionError(f"No flagged noise hit found in the first {len(events)} event(s) from {source_dir}.")

        event = noisy_events[0]
        apply_variable_count_target_mode(
            event,
            valid_labels=VALID_LABELS,
            target_mode="canonical-y",
            max_electrons=len(VALID_LABELS),
        )
        noise_mask = event["is_noise_target"]
        self.assertTrue(bool((event["physical_y"][noise_mask] == 0).all().item()))
        self.assertTrue(bool((event["canonical_y"][noise_mask] == -1).all().item()))
        self.assertEqual(event["target_label_order"], _canonical_order_from_non_noise_hits(event))

        model = ECalTpadSlotModel(
            in_dim=int(event["x"].shape[1]),
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            max_electrons=len(VALID_LABELS),
            dropout=0.0,
            use_type_embedding=True,
        ).cpu()
        args = _loss_args()
        losses = compute_event_losses(model, event, torch.device("cpu"), args)
        expected_noise_fraction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        expected_rows = expected_noise_fraction.unsqueeze(0).expand(int(noise_mask.sum().item()), -1)
        self.assertTrue(torch.equal(losses["fraction_target"][noise_mask].cpu(), expected_rows))
        for key in ("total_loss", "origin_loss", "fraction_loss", "slot_loss", "count_loss"):
            self.assertTrue(bool(torch.isfinite(losses[key]).item()), f"{key} is not finite")
        losses["total_loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

        metrics, _predictions = evaluate(model, [event], [0], args, torch.device("cpu"), "noise")
        background_rows = sum(metrics["noise_hit_confusion"][0])
        self.assertEqual(background_rows, int(noise_mask.sum().item()))


if __name__ == "__main__":
    unittest.main()
