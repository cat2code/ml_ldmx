"""Optional ROOT-backed smoke test for sharded preprocessing and lazy access."""

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from ml_ldmx.datasets.ecal_tpad_loading import apply_variable_count_target_mode, filter_noise_tensor_event
from ml_ldmx.datasets.ecal_tpad_shards import (
    ShardedECalTpadDataset,
    prepare_sharded_tensor_cache,
    validate_sharded_tensor_cache,
)
from ml_ldmx.datasets.model_views import ecal_transformer_view
from ml_ldmx.datasets.tensorize import transform_ecal_energy, transform_tpad_pe
from ml_ldmx.models import ECalTransformer
from ml_ldmx.train.hit_classifier_baseline import compute_event_losses


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
        if (data_root / "2e/events").exists() and (data_root / "3e/events").exists():
            return data_root
    searched = ", ".join(str(path) for path in candidates)
    raise unittest.SkipTest(f"No complete 2e/3e ROOT data directory found. Set ML_LDMX_ROOT_DATA. Searched: {searched}")


def _root_specs():
    root_2e = os.environ.get("ML_LDMX_ROOT_2E_DIR")
    root_3e = os.environ.get("ML_LDMX_ROOT_3E_DIR")
    if root_2e and root_3e:
        return [(2, "2e", Path(root_2e)), (3, "3e", Path(root_3e))]
    data_root = _root_data_root()
    return [
        (2, "2e", data_root / "2e/events"),
        (3, "3e", data_root / "3e/events"),
    ]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _smoke_device() -> torch.device:
    requested = os.environ.get("ML_LDMX_SMOKE_DEVICE", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise unittest.SkipTest("ML_LDMX_SMOKE_DEVICE=cuda requested, but CUDA is unavailable.")
    return torch.device(requested)


def _canonical_filtered_transform(event: dict) -> dict:
    event = filter_noise_tensor_event(event)
    return apply_variable_count_target_mode(
        event,
        valid_labels=VALID_LABELS,
        target_mode="canonical-y",
        max_electrons=len(VALID_LABELS),
    )


def _canonical_noise_transform(event: dict) -> dict:
    return apply_variable_count_target_mode(
        event,
        valid_labels=VALID_LABELS,
        target_mode="canonical-y",
        max_electrons=len(VALID_LABELS),
    )


class ShardedCacheSmokeTest(unittest.TestCase):
    def test_root_to_sharded_cache_lazy_reuse_and_training_step(self):
        torch.manual_seed(7)
        max_root_files = _env_int("ML_LDMX_SHARD_SMOKE_MAX_ROOT_FILES", 1)
        max_events_per_root_file = _env_int("ML_LDMX_SHARD_SMOKE_MAX_EVENTS_PER_ROOT_FILE", 10)
        ecal_energy_transform = os.environ.get("ML_LDMX_ECAL_ENERGY_TRANSFORM", "raw")
        tpad_pe_transform = os.environ.get("ML_LDMX_TPAD_PE_TRANSFORM", "raw")
        root_specs = _root_specs()

        with TemporaryDirectory(prefix="ml_ldmx_sharded_smoke_") as temporary_dir:
            cache_dir = Path(temporary_dir)
            common_kwargs = {
                "cache_dir": cache_dir,
                "root_specs": root_specs,
                "valid_labels": VALID_LABELS,
                "filter_noise": False,
                "supervise_noise": True,
                "max_root_files": max_root_files,
                "max_events_per_root_file": max_events_per_root_file,
                "read_step_size": 50,
                "ecal_energy_transform": ecal_energy_transform,
                "tpad_pe_transform": tpad_pe_transform,
                "logger": logging.getLogger("sharded-cache-smoke-test"),
            }
            prepare_sharded_tensor_cache(**common_kwargs)
            prepare_sharded_tensor_cache(**common_kwargs)
            manifest, index = validate_sharded_tensor_cache(cache_dir, load_shards=True)
            self.assertEqual(manifest["ecal_energy_transform"], ecal_energy_transform)
            self.assertEqual(manifest["tpad_pe_transform"], tpad_pe_transform)
            dataset = ShardedECalTpadDataset(cache_dir, shard_cache_size=1)
            expected_events = sum(entry["num_events"] for entry in index["shards"])
            self.assertEqual(len(dataset), expected_events)
            self.assertGreaterEqual(len(index["shards"]), 2)

            stored_event = dataset[0]
            self.assertIn("ecal_raw_energy", stored_event)
            self.assertIn("tpad_raw_pe", stored_event)
            self.assertTrue(
                torch.allclose(
                    stored_event["ecal_input_energy"],
                    transform_ecal_energy(stored_event["ecal_raw_energy"], ecal_energy_transform),
                )
            )
            self.assertTrue(
                torch.allclose(
                    stored_event["tpad"][:, 1],
                    transform_tpad_pe(stored_event["tpad_raw_pe"], tpad_pe_transform),
                )
            )

            source_transition_idx = next(
                (
                    shard_idx
                    for shard_idx in range(1, len(index["shards"]))
                    if index["shards"][shard_idx - 1]["source"]["source_label"]
                    != index["shards"][shard_idx]["source"]["source_label"]
                ),
                None,
            )
            self.assertIsNotNone(source_transition_idx)
            source_boundary_start = index["shards"][source_transition_idx]["event_start"]
            self.assertEqual(int(dataset[source_boundary_start]["event_idx"]), source_boundary_start)
            self.assertNotEqual(
                dataset[source_boundary_start - 1]["source_label"],
                dataset[source_boundary_start]["source_label"],
            )
            if max_root_files >= 2:
                first_two_names = [entry["source"]["name"] for entry in index["shards"][:2]]
                self.assertEqual(first_two_names, ["events_1.root", "events_2.root"])

            raw_noise_hits = sum(int(dataset[index]["is_noise_target"].sum().item()) for index in range(len(dataset)))
            self.assertGreater(raw_noise_hits, 0)

            device = _smoke_device()
            dataset.set_event_transform(_canonical_filtered_transform)
            view = ecal_transformer_view(dataset[0])
            model = ECalTransformer(
                in_dim=int(view["x"].shape[1]),
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
                out_dim=len(VALID_LABELS),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            losses = compute_event_losses(model, dataset[0], ecal_transformer_view, device)
            self.assertTrue(bool(torch.isfinite(losses["total_loss"]).item()))
            losses["total_loss"].backward()
            optimizer.step()

            dataset.set_event_transform(_canonical_noise_transform)
            supervised_noise_hits = sum(
                int(dataset[index]["is_noise_target"].sum().item()) for index in range(len(dataset))
            )
            self.assertEqual(supervised_noise_hits, raw_noise_hits)


if __name__ == "__main__":
    unittest.main()
