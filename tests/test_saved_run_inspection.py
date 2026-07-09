import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from inspect_hit_classifier_run import (
    resolve_checkpoint_path,
    restore_event_preprocessing,
    summarize_records,
)


class SavedRunInspectionTest(unittest.TestCase):
    def test_checkpoint_resolution_prefers_best_then_latest(self):
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            latest_path = checkpoint_dir / "latest.pt"
            latest_path.touch()

            self.assertEqual(resolve_checkpoint_path(run_dir), latest_path.resolve())

            best_path = checkpoint_dir / "best.pt"
            best_path.touch()
            self.assertEqual(resolve_checkpoint_path(run_dir), best_path.resolve())
            self.assertEqual(
                resolve_checkpoint_path(run_dir, Path("latest.pt")),
                latest_path.resolve(),
            )

    def test_restore_event_preprocessing_uses_checkpoint_normalization(self):
        event = {
            "x": torch.tensor(
                [
                    [1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                    [1.0, 0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
                ]
            ),
            "ecal_mask": torch.tensor([True, True]),
            "tpad_mask": torch.tensor([False, False]),
            "ecal_pos": torch.tensor([[0.0, -5.0, 10.0], [0.0, 5.0, 10.0]]),
            "physical_y": torch.tensor([2, 1]),
            "y": torch.tensor([2, 1]),
            "ecal_input_energy": torch.tensor([6.0, 8.0]),
            "event_idx": 0,
        }
        checkpoint = {
            "feature_norm": {
                "first_continuous_col": 2,
                "mean": [0.0] * 6,
                "std": [2.0] * 6,
            }
        }
        args = SimpleNamespace(valid_labels=[1, 2], target_mode="canonical-y")
        original_continuous = event["x"][:, 2:].clone()

        restored = restore_event_preprocessing([event], checkpoint, args)[0]

        torch.testing.assert_close(
            restored["x"][:, 2:],
            original_continuous / 2.0,
        )
        self.assertEqual(restored["target_label_order"], [2, 1])
        self.assertEqual(restored["y"].tolist(), [0, 1])
        self.assertEqual(restored["ecal_input_energy"].tolist(), [6.0, 8.0])

    def test_summarize_records_reports_hit_and_event_aggregates(self):
        summary = summarize_records(
            [
                {"num_hits": 4, "correct_hits": 3, "accuracy": 0.75, "loss": 0.4},
                {"num_hits": 6, "correct_hits": 3, "accuracy": 0.5, "loss": 0.8},
            ]
        )

        self.assertEqual(summary["num_events"], 2)
        self.assertEqual(summary["num_hits"], 10)
        self.assertAlmostEqual(summary["hit_accuracy"], 0.6)
        self.assertAlmostEqual(summary["mean_event_accuracy"], 0.625)
        self.assertAlmostEqual(summary["median_event_accuracy"], 0.625)
        self.assertAlmostEqual(summary["mean_event_loss"], 0.6)


if __name__ == "__main__":
    unittest.main()
