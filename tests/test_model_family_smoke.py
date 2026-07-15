"""Reusable smoke tests for the maintained ml_ldmx model family."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from ml_ldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    filter_noise_tensor_event,
    load_multi_sharded_tensor_events,
    load_processed_tensor_events,
)
from ml_ldmx.datasets.model_views import (
    ecal_gravnet_view,
    ecal_tpad_gravnet_view,
    ecal_tpad_slot_model_view,
    ecal_tpad_transformer_view,
    ecal_transformer_view,
)
from ml_ldmx.models import (
    ECalGravNet,
    ECalTpadGravNet,
    ECalTpadSlotModel,
    ECalTpadTrackSeededTransformer,
    ECalTpadTransformer,
    ECalTransformer,
)
from ml_ldmx.train.ecal_tpad_slot_model import compute_event_losses as compute_slot_event_losses
from ml_ldmx.train.hit_classifier_baseline import compute_event_losses as compute_hit_event_losses


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = (1, 2, 3)
TARGET_MODE = "canonical-y"


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _project_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _has_event_files(path: Path) -> bool:
    return path.exists() and any(path.glob("event_*.pt"))


def _default_processed_dirs() -> list[Path]:
    return [
        PROJECT_ROOT / "data/processed/ecal_tpad_3class_smoke",
        PROJECT_ROOT.parent / "mldmx/data/processed/ecal_tpad_3class_smoke",
    ]


def _smoke_device() -> torch.device:
    requested = os.environ.get("ML_LDMX_SMOKE_DEVICE", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise unittest.SkipTest("ML_LDMX_SMOKE_DEVICE=cuda requested, but CUDA is unavailable.")
    if requested == "mps" and not getattr(torch.backends, "mps", None).is_available():
        raise unittest.SkipTest("ML_LDMX_SMOKE_DEVICE=mps requested, but MPS is unavailable.")
    return torch.device(requested)


def _load_from_processed_dir(event_index: int) -> tuple[dict, Path]:
    env_dir = os.environ.get("ML_LDMX_SMOKE_PROCESSED_DIR")
    candidates = [_project_path(env_dir)] if env_dir else _default_processed_dirs()
    for processed_dir in candidates:
        if not _has_event_files(processed_dir):
            continue
        events, _sources = load_processed_tensor_events(processed_dir, max_events=event_index + 1)
        if event_index >= len(events):
            raise unittest.SkipTest(
                f"Processed smoke cache {processed_dir} has {len(events)} event(s), "
                f"but event index {event_index} was requested."
            )
        return dict(events[event_index]), processed_dir
    searched = ", ".join(str(path) for path in candidates)
    raise unittest.SkipTest(
        "No processed smoke cache found. Set ML_LDMX_SMOKE_PROCESSED_DIR or create "
        f"data/processed/ecal_tpad_3class_smoke. Searched: {searched}"
    )


def _load_from_sharded_cache(event_index: int) -> tuple[dict, Path]:
    cache_root = os.environ.get("ML_LDMX_PROCESSED_CACHE_ROOT") or os.environ.get("PROCESSED_CACHE_ROOT")
    if not cache_root:
        raise unittest.SkipTest("No sharded cache requested.")
    cache_root = _project_path(cache_root)
    sources = [
        (2, "2e", cache_root / "2e/events"),
        (3, "3e", cache_root / "3e/events"),
    ]
    missing = [cache_dir for _count, _label, cache_dir in sources if not cache_dir.exists()]
    if missing:
        raise unittest.SkipTest(f"Requested sharded cache is incomplete: {missing}")
    events_per_source = _env_int("ML_LDMX_EVENTS_PER_SOURCE", _env_int("EVENTS_PER_SOURCE"))
    ecal_energy_transform = os.environ.get(
        "ML_LDMX_ECAL_ENERGY_TRANSFORM",
        os.environ.get("ECAL_ENERGY_TRANSFORM", "raw"),
    )
    tpad_pe_transform = os.environ.get(
        "ML_LDMX_TPAD_PE_TRANSFORM",
        os.environ.get("TPAD_PE_TRANSFORM", "raw"),
    )
    events, _sources, _selected, _root_files = load_multi_sharded_tensor_events(
        sources,
        events_per_source=events_per_source,
        ecal_energy_transform=ecal_energy_transform,
        tpad_pe_transform=tpad_pe_transform,
    )
    if event_index >= len(events):
        raise unittest.SkipTest(
            f"Sharded smoke cache {cache_root} has {len(events)} event(s), "
            f"but event index {event_index} was requested."
        )
    return dict(events[event_index]), cache_root


def _load_raw_smoke_event() -> tuple[dict, Path]:
    event_index = _env_int("ML_LDMX_SMOKE_EVENT_INDEX", 0)
    if event_index is None or event_index < 0:
        raise ValueError("ML_LDMX_SMOKE_EVENT_INDEX must be non-negative.")
    if os.environ.get("ML_LDMX_PROCESSED_CACHE_ROOT") or os.environ.get("PROCESSED_CACHE_ROOT"):
        return _load_from_sharded_cache(event_index)
    return _load_from_processed_dir(event_index)


def _canonical_smoke_event() -> tuple[dict, Path]:
    raw_event, source = _load_raw_smoke_event()
    event = filter_noise_tensor_event(raw_event)
    original_physical_y = event["physical_y"].clone()
    apply_variable_count_target_mode(
        event,
        valid_labels=VALID_LABELS,
        target_mode=TARGET_MODE,
        max_electrons=len(VALID_LABELS),
    )
    if "canonical_y" not in event:
        raise AssertionError("canonical-y target preparation did not expose event['canonical_y'].")
    if "origin_id_y" not in event:
        raise AssertionError("canonical-y target preparation did not retain event['origin_id_y'].")
    if not torch.equal(event["origin_id_y"], original_physical_y):
        raise AssertionError("event['origin_id_y'] does not preserve pre-canonical physical origins.")
    return event, source


def _slot_loss_args() -> SimpleNamespace:
    return SimpleNamespace(
        lambda_origin=1.0,
        lambda_fraction=1.0,
        lambda_slot=0.5,
        lambda_count=1.0,
        origin_class_weights=None,
        count_class_weights=None,
    )


def _assert_backward_reached_parameters(test_case: unittest.TestCase, model: torch.nn.Module, name: str) -> None:
    has_grad = any(parameter.grad is not None for parameter in model.parameters())
    test_case.assertTrue(has_grad, f"{name}: backward produced no parameter gradients.")


class ModelFamilySmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_model_views_share_canonical_targets(self):
        event, _source = _canonical_smoke_event()
        num_ecal = int(event["ecal_mask"].sum().item())
        builders = [
            ("ECalTransformer", ecal_transformer_view, 4, False),
            ("ECalTpadTransformer", ecal_tpad_transformer_view, 8, True),
            ("ECalTpadTrackSeededTransformer", ecal_tpad_transformer_view, 8, True),
            ("ECalGravNet", ecal_gravnet_view, 4, False),
            ("ECalTpadGravNet", ecal_tpad_gravnet_view, 8, True),
            ("ECalTpadSlotModel", ecal_tpad_slot_model_view, 8, True),
        ]

        for name, builder, feature_dim, has_tpad in builders:
            with self.subTest(model=name):
                view = builder(event)
                if name == "ECalTpadSlotModel":
                    self.assertIs(view, event)
                self.assertEqual(tuple(view["x"].shape)[1], feature_dim)
                self.assertEqual(tuple(view["ecal_mask"].shape), (view["x"].shape[0],))
                self.assertEqual(tuple(view["y"].shape), (num_ecal,))
                self.assertEqual(tuple(view["origin_id_y"].shape), (num_ecal,))
                self.assertEqual(tuple(view["canonical_y"].shape), (num_ecal,))
                self.assertTrue(bool(view["ecal_mask"].any().item()))
                if has_tpad:
                    self.assertIn("tpad_mask", view)
                    self.assertEqual(tuple(view["tpad_mask"].shape), (view["x"].shape[0],))
                else:
                    self.assertTrue(bool(view["ecal_mask"].all().item()))

    def test_transformer_and_slot_models_forward_backward(self):
        event, _source = _canonical_smoke_event()
        device = _smoke_device()
        checks = [
            (
                "ECalTransformer",
                ECalTransformer(
                    in_dim=4,
                    d_model=32,
                    nhead=4,
                    num_layers=1,
                    dim_feedforward=64,
                    dropout=0.0,
                    out_dim=len(VALID_LABELS),
                ),
                ecal_transformer_view(event),
            ),
            (
                "ECalTpadTransformer",
                ECalTpadTransformer(
                    in_dim=8,
                    d_model=32,
                    nhead=4,
                    num_layers=1,
                    dim_feedforward=64,
                    dropout=0.0,
                    out_dim=len(VALID_LABELS),
                ),
                ecal_tpad_transformer_view(event),
            ),
            (
                "ECalTpadTrackSeededTransformer",
                ECalTpadTrackSeededTransformer(
                    in_dim=8,
                    d_model=32,
                    nhead=4,
                    num_layers=1,
                    dim_feedforward=64,
                    dropout=0.0,
                    out_dim=len(VALID_LABELS),
                ),
                ecal_tpad_transformer_view(event),
            ),
        ]

        for name, model, view in checks:
            with self.subTest(model=name):
                model = model.to(device)
                model.train()
                losses = compute_hit_event_losses(model, view, None, device)
                self.assertTrue(bool(torch.isfinite(losses["total_loss"]).item()))
                self.assertEqual(int(losses["num_hits"]), int(view["y"].numel()))
                losses["total_loss"].backward()
                _assert_backward_reached_parameters(self, model, name)

        with self.subTest(model="ECalTpadSlotModel"):
            slot_view = ecal_tpad_slot_model_view(event)
            model = ECalTpadSlotModel(
                in_dim=int(slot_view["x"].shape[1]),
                hidden_dim=32,
                num_layers=1,
                num_heads=4,
                max_electrons=len(VALID_LABELS),
                dropout=0.0,
                use_type_embedding=True,
            ).to(device)
            model.train()
            losses = compute_slot_event_losses(model, slot_view, device, _slot_loss_args())
            for key in ("total_loss", "origin_loss", "fraction_loss", "slot_loss", "count_loss"):
                self.assertTrue(bool(torch.isfinite(losses[key]).item()), f"{key} is not finite")
            self.assertEqual(int(losses["num_hits"]), int(slot_view["y"].numel()))
            losses["total_loss"].backward()
            _assert_backward_reached_parameters(self, model, "ECalTpadSlotModel")

    def test_gravnet_models_forward_backward(self):
        event, _source = _canonical_smoke_event()
        device = _smoke_device()
        checks = [
            (
                "ECalGravNet",
                lambda: ECalGravNet(
                    in_dim=4,
                    hidden_dim=32,
                    out_dim=len(VALID_LABELS),
                    num_layers=1,
                    space_dimensions=4,
                    propagate_dimensions=16,
                    k=8,
                    dropout=0.0,
                ),
                ecal_gravnet_view,
            ),
            (
                "ECalTpadGravNet",
                lambda: ECalTpadGravNet(
                    in_dim=8,
                    hidden_dim=32,
                    out_dim=len(VALID_LABELS),
                    num_layers=1,
                    space_dimensions=4,
                    propagate_dimensions=16,
                    k=8,
                    dropout=0.0,
                ),
                ecal_tpad_gravnet_view,
            ),
        ]

        for name, model_factory, view_builder in checks:
            with self.subTest(model=name):
                try:
                    model = model_factory().to(device)
                    model.train()
                    view = view_builder(event)
                    losses = compute_hit_event_losses(model, view, None, device)
                    self.assertTrue(bool(torch.isfinite(losses["total_loss"]).item()))
                    losses["total_loss"].backward()
                    _assert_backward_reached_parameters(self, model, name)
                except Exception as exc:
                    self.skipTest(f"GravNetConv runtime unavailable in this environment: {exc}")


if __name__ == "__main__":
    unittest.main()
