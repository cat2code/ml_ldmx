from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.nn as nn

from ml_ldmx.eval.event_diagnostics import select_representative_events
from ml_ldmx.eval.hit_classifier_baseline import collect_event_metrics
from ml_ldmx.viz.training import plot_event_accuracy_overview, plot_event_diagnostic_correlations


class IdentityLogitModel(nn.Module):
    def forward(self, x):
        return x


def _view(logits, target, event_idx, pos=None, energy=None):
    num_hits = len(target)
    if pos is None:
        pos = [[float(idx), float(label), float(idx % 2)] for idx, label in enumerate(target)]
    if energy is None:
        energy = [1.0 for _idx in range(num_hits)]
    return {
        "x": torch.as_tensor(logits, dtype=torch.float32),
        "ecal_mask": torch.ones((num_hits,), dtype=torch.bool),
        "ecal_pos": torch.as_tensor(pos, dtype=torch.float32),
        "ecal_input_energy": torch.as_tensor(energy, dtype=torch.float32),
        "ecal_raw_energy": torch.as_tensor(energy, dtype=torch.float32),
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
                pos=[[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [40.0, 0.0, 10.0]],
                energy=[10.0, 1.0, 1.0],
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
        self.assertAlmostEqual(records[0]["energy_weighted_accuracy"], 11 / 12)
        self.assertGreater(records[0]["mean_confidence"], 0.0)
        self.assertEqual(records[0]["num_truth_classes"], 3)
        self.assertAlmostEqual(records[0]["min_origin_centroid_distance_xy"], 10.0)
        self.assertEqual(records[1]["correct_hits"], 2)
        self.assertEqual(records[1]["incorrect_hits"], 0)
        self.assertEqual(records[1]["accuracy"], 1.0)
        self.assertEqual(records[1]["source_file"], "event_11.root")
        self.assertEqual(records[1]["source_entry"], 11)

    def test_normalized_shower_separation_and_ambiguity_metrics(self):
        events = [
            _view(
                [[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]],
                [0, 0, 1, 1],
                12,
                pos=[
                    [-1.0, 0.0, 10.0],
                    [1.0, 0.0, 10.0],
                    [9.0, 0.0, 10.0],
                    [11.0, 0.0, 10.0],
                ],
            )
        ]
        args = SimpleNamespace(batch_size=1, valid_labels=[0, 1])

        record = collect_event_metrics(
            IdentityLogitModel(),
            events,
            [0],
            None,
            args,
            torch.device("cpu"),
        )[0]

        self.assertAlmostEqual(
            record["min_normalized_shower_separation_xy"],
            10.0 / (2.0**0.5),
        )
        self.assertAlmostEqual(
            record["early_min_normalized_shower_separation_xy"],
            10.0 / (2.0**0.5),
        )
        self.assertEqual(record["ambiguous_hit_fraction_xy"], 0.0)
        self.assertEqual(record["early_layer_count"], 1)
        self.assertEqual(record["shower_overlap_weighting"], "raw_reconstructed_energy")

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

    def test_plot_event_diagnostic_correlations_writes_file(self):
        records = [
            {
                "event_idx": 12,
                "split_position": 0,
                "num_hits": 100,
                "incorrect_hits": 5,
                "accuracy": 0.95,
                "loss": 0.1,
                "mean_confidence": 0.88,
                "min_origin_centroid_distance_xy": 55.0,
            },
            {
                "event_idx": 15,
                "split_position": 1,
                "num_hits": 120,
                "incorrect_hits": 48,
                "accuracy": 0.60,
                "loss": 1.4,
                "mean_confidence": 0.62,
                "min_origin_centroid_distance_xy": 12.0,
            },
            {
                "event_idx": 20,
                "split_position": 2,
                "num_hits": 80,
                "incorrect_hits": 12,
                "accuracy": 0.85,
                "loss": 0.4,
                "mean_confidence": 0.75,
                "min_origin_centroid_distance_xy": 30.0,
            },
        ]
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "event_diagnostics.png"
            plot_event_diagnostic_correlations(records, output_path, "validation diagnostics")

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_select_representative_events_returns_worst_median_best(self):
        records = [
            {"event_idx": 1, "accuracy": 0.2, "incorrect_hits": 8},
            {"event_idx": 2, "accuracy": 0.6, "incorrect_hits": 4},
            {"event_idx": 3, "accuracy": 0.9, "incorrect_hits": 1},
        ]

        selection = select_representative_events(records, limit_per_group=1)

        self.assertEqual(selection["worst"][0]["event_idx"], 1)
        self.assertEqual(selection["median"][0]["event_idx"], 2)
        self.assertEqual(selection["best"][0]["event_idx"], 3)


if __name__ == "__main__":
    unittest.main()
