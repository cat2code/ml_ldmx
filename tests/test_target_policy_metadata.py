import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from inspect_hit_classifier_run import _training_args, restore_model
from ml_ldmx.datasets.ecal_tpad_shards import (
    _cache_spec,
    _cache_specs_compatible_for_loading,
    _cache_specs_match,
)
from ml_ldmx.datasets.tensorize import (
    DOMINANT_ORIGIN_TARGET_RULE,
    LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
)
from ml_ldmx.models import ECalGravNet
from ml_ldmx.train.checkpoints import (
    checkpoint_hard_origin_target_rule,
    checkpoint_state,
    require_matching_hard_origin_target_rule,
)


def _inspection_args():
    return SimpleNamespace(
        processed_dir=None,
        processed_cache=None,
        processed_cache_root=None,
        processed_source=None,
        data_root=None,
        events_per_source=None,
        shard_cache_size=None,
        batch_size=None,
        event_diagnostic_radius_mm=None,
        evaluation_hard_origin_target_rule=None,
        num_events=3,
    )


class TargetPolicyMetadataTest(unittest.TestCase):
    def test_inspector_restores_legacy_gravnet_without_batch_normalization(self):
        legacy_kwargs = {
            "in_dim": 3,
            "hidden_dim": 4,
            "out_dim": 3,
            "num_layers": 1,
            "space_dimensions": 2,
            "propagate_dimensions": 4,
            "k": 2,
            "dropout": 0.0,
        }
        try:
            legacy_model = ECalGravNet(**legacy_kwargs, normalization="none")
        except Exception as exc:
            self.skipTest(f"GravNetConv runtime unavailable in this environment: {exc}")
        checkpoint = {
            "model_kwargs": legacy_kwargs,
            "model_state_dict": legacy_model.state_dict(),
        }

        restored, _view = restore_model(
            checkpoint,
            SimpleNamespace(model="ECalGravNet"),
            torch.device("cpu"),
        )

        self.assertEqual(restored.normalization, "none")
        self.assertTrue(all(isinstance(norm, torch.nn.Identity) for norm in restored.norms))

    def test_old_checkpoint_defaults_to_legacy_rule(self):
        checkpoint = {"args": {}}

        self.assertEqual(
            checkpoint_hard_origin_target_rule(checkpoint),
            LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
        )
        args = _training_args(
            checkpoint,
            {"hard_origin_target_rule": DOMINANT_ORIGIN_TARGET_RULE},
            _inspection_args(),
        )
        self.assertEqual(
            args.hard_origin_target_rule,
            LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
        )

    def test_resume_rejects_target_rule_change(self):
        checkpoint = {"args": {}}

        with self.assertRaisesRegex(ValueError, "hard-origin target rule"):
            require_matching_hard_origin_target_rule(
                checkpoint,
                DOMINANT_ORIGIN_TARGET_RULE,
            )

    def test_inspector_can_override_rule_for_cross_target_evaluation(self):
        checkpoint = {
            "hard_origin_target_rule": LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
            "args": {},
        }
        inspection_args = _inspection_args()
        inspection_args.evaluation_hard_origin_target_rule = (
            DOMINANT_ORIGIN_TARGET_RULE
        )

        args = _training_args(checkpoint, {}, inspection_args)

        self.assertEqual(
            args.hard_origin_target_rule,
            DOMINANT_ORIGIN_TARGET_RULE,
        )

    def test_new_checkpoint_records_rule_at_both_metadata_levels(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        args = SimpleNamespace(
            valid_labels=[1, 2],
            hard_origin_target_rule=DOMINANT_ORIGIN_TARGET_RULE,
        )

        state = checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=0,
            args=args,
            history=[],
            best_val_loss=1.0,
            model_kwargs={},
            feature_norm=None,
            splits={"train": [0], "val": [], "test": []},
        )

        self.assertEqual(
            state["hard_origin_target_rule"],
            DOMINANT_ORIGIN_TARGET_RULE,
        )
        self.assertEqual(
            state["args"]["hard_origin_target_rule"],
            DOMINANT_ORIGIN_TARGET_RULE,
        )

    def test_legacy_and_new_cache_specs_do_not_match(self):
        common = {
            "root_sources": [],
            "valid_labels": (1, 2, 3),
            "filter_noise": False,
            "supervise_noise": True,
            "max_events_per_root_file": None,
        }
        new_spec = _cache_spec(**common)
        old_spec = dict(new_spec)
        old_spec.pop("hard_origin_target_rule")

        self.assertFalse(_cache_specs_match(old_spec, new_spec))
        self.assertEqual(
            new_spec["hard_origin_target_rule"],
            DOMINANT_ORIGIN_TARGET_RULE,
        )
        self.assertTrue(_cache_specs_compatible_for_loading(old_spec, new_spec))
        self.assertFalse(_cache_specs_compatible_for_loading(new_spec, old_spec))


if __name__ == "__main__":
    unittest.main()
