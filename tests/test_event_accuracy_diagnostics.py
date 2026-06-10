from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.nn as nn

from mldmx.eval.hit_classifier_baseline import collect_event_metrics
from mldmx.viz.training import plot_event_accuracy_overview


class IdentityLogitModel(nn.Module):
    def forward(self, x):
        return x


def _view(logits, target, event_idx):
    num_hits = len(target)
    return {
        "x": torch.as_tensor(logits, dtype=torch.float32),
        "ecal_mask": torch.ones((num_hits,), dtype=torch.bool),
        "y": torch.as_tensor(target, dtype=torch.long),
        "source_file": f"event_{event_idx}.root",
        "source_entry": torch.tensor(event_idx),
    }


class EventAccuracyDiagnosticsTest(unittest.TestCase):
    def test_collect_event_metrics_reports_per_event_accuracy(self):
        events = [
            _view(
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]],
                [0, 1, 2],
                10,
            ),
            _view(
                [[0.0, 0.0, 5.0], [0.0, 5.0, 0.0]],
                [2, 1],
                11,
            ),
        ]
        args = SimpleNamespace(batch_size=2, valid_labels=[0, 1, 2])

        records = collect_event_metrics(
            IdentityLogitModel(),
            events,
            [0, 1],
            None,
            args,
            torch.device("cpu"),
        )

        self.assertEqual([record["event_idx"] for record in records], [0, 1])
        self.assertEqual(records[0]["correct_hits"], 2)
        self.assertEqual(records[0]["incorrect_hits"], 1)
        self.assertAlmostEqual(records[0]["accuracy"], 2 / 3)
        self.assertEqual(records[1]["correct_hits"], 2)
        self.assertEqual(records[1]["incorrect_hits"], 0)
        self.assertEqual(records[1]["accuracy"], 1.0)
        self.assertEqual(records[1]["source_file"], "event_11.root")
        self.assertEqual(records[1]["source_entry"], 11)

    def test_plot_event_accuracy_overview_writes_file(self):
        records = [
            {
                "event_idx": 12,
                "split_position": 0,
                "num_hits": 100,
                "correct_hits": 95,
                "incorrect_hits": 5,
                "accuracy": 0.95,
            },
            {
                "event_idx": 15,
                "split_position": 1,
                "num_hits": 120,
                "correct_hits": 72,
                "incorrect_hits": 48,
                "accuracy": 0.60,
            },
        ]
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "event_accuracy.png"
            plot_event_accuracy_overview(records, output_path, "validation accuracy")

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
