import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_contributor_set_tpad_ablation import (  # noqa: E402
    TPadAblatedDataset,
    remove_tpad_tokens,
    summarize_ablation,
)
from ml_ldmx.viz.contributor_set_ablation import (  # noqa: E402
    plot_confusion_ablation,
    plot_count_ablation,
    plot_fraction_mae_ablation,
    plot_task_metric_ablation,
)


class ContributorSetTPadAblationTest(unittest.TestCase):
    def test_removing_tpad_tokens_preserves_ecal_inputs_and_targets(self):
        event = {
            "x": torch.arange(20, dtype=torch.float32).reshape(5, 4),
            "ecal_mask": torch.tensor([True, False, True, False, True]),
            "tpad_mask": torch.tensor([False, True, False, True, False]),
            "tpad": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "tpad_raw_pe": torch.tensor([4.0, 8.0]),
            "fraction_target": torch.eye(4)[:3],
            "electron_count": 3,
        }

        ablated = remove_tpad_tokens(event)

        self.assertTrue(torch.equal(ablated["x"], event["x"][[0, 2, 4]]))
        self.assertEqual(ablated["ecal_mask"].tolist(), [True, True, True])
        self.assertEqual(ablated["tpad_mask"].tolist(), [False, False, False])
        self.assertEqual(ablated["tpad"].shape[0], 0)
        self.assertTrue(torch.equal(ablated["fraction_target"], event["fraction_target"]))
        self.assertEqual(ablated["electron_count"], 3)
        self.assertEqual(event["x"].shape[0], 5)

    def test_dataset_view_delegates_saved_access_order(self):
        class Events:
            def __init__(self, event):
                self.event = event

            def __len__(self):
                return 1

            def __getitem__(self, index):
                self.assert_index = index
                return self.event

            def order_indices_for_access(self, indices, seed=None):
                return list(reversed(indices))

        event = {
            "x": torch.ones((2, 4)),
            "ecal_mask": torch.tensor([True, False]),
            "tpad_mask": torch.tensor([False, True]),
        }
        ablated = TPadAblatedDataset(Events(event))
        self.assertEqual(ablated.order_indices_for_access([0, 1]), [1, 0])
        self.assertEqual(ablated[0]["x"].shape[0], 1)

    def test_summary_uses_positive_values_for_helpful_tpad_effects(self):
        reference = {
            "count_accuracy": 0.90,
            "support_accuracy": 0.70,
            "fraction_mae": 0.10,
            "loss": 1.2,
        }
        removed = {
            "count_accuracy": 0.65,
            "support_accuracy": 0.60,
            "fraction_mae": 0.14,
            "loss": 1.7,
        }

        summary = summarize_ablation(reference, removed)
        gain = summary["tpad_gain_positive_is_better"]

        self.assertAlmostEqual(gain["count_accuracy"], 0.25)
        self.assertAlmostEqual(gain["support_accuracy"], 0.10)
        self.assertAlmostEqual(gain["fraction_mae_reduction"], 0.04)
        self.assertAlmostEqual(gain["loss_reduction"], 0.5)

    def test_all_paired_plots_are_written(self):
        reference = {
            "count_accuracy": 0.75,
            "count_accuracy_2e": 1.0,
            "count_accuracy_3e": 0.5,
            "slot_exact_accuracy": 0.75,
            "support_accuracy": 0.70,
            "mixed_f1": 0.60,
            "origin_accuracy": 0.72,
            "fraction_mae": 0.10,
            "raw_fraction_mae": 0.12,
            "mixed_brier": 0.15,
        }
        removed = {
            "count_accuracy": 0.50,
            "count_accuracy_2e": 1.0,
            "count_accuracy_3e": 0.0,
            "slot_exact_accuracy": 0.50,
            "support_accuracy": 0.60,
            "mixed_f1": 0.40,
            "origin_accuracy": 0.62,
            "fraction_mae": 0.15,
            "raw_fraction_mae": 0.17,
            "mixed_brier": 0.25,
        }
        reference_predictions = [
            {
                "event_index": 10,
                "true_count": 2,
                "predicted_count": 2,
                "slot_probability": [0.99, 0.99, 0.10],
            },
            {
                "event_index": 20,
                "true_count": 3,
                "predicted_count": 3,
                "slot_probability": [0.99, 0.99, 0.90],
            },
        ]
        removed_predictions = [
            {
                "event_index": 20,
                "true_count": 3,
                "predicted_count": 2,
                "slot_probability": [0.99, 0.99, 0.30],
            },
            {
                "event_index": 10,
                "true_count": 2,
                "predicted_count": 2,
                "slot_probability": [0.99, 0.99, 0.15],
            },
        ]
        target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.2, 0.3, 0.5]]
        )
        reference_plot_data = {
            "fraction_target": target,
            "fraction_pred": torch.tensor(
                [[0.0, 0.9, 0.1, 0.0], [0.0, 0.2, 0.4, 0.4]]
            ),
        }
        removed_plot_data = {
            "fraction_target": target.clone(),
            "fraction_pred": torch.tensor(
                [[0.0, 0.7, 0.3, 0.0], [0.0, 0.4, 0.4, 0.2]]
            ),
        }

        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            paths = [
                output_dir / "confusion.png",
                output_dir / "metrics.png",
                output_dir / "count.png",
                output_dir / "fractions.png",
            ]
            plot_confusion_ablation(
                [[8, 2], [1, 9]],
                [[6, 4], [5, 5]],
                ["2e", "3e"],
                paths[0],
                "test",
            )
            plot_task_metric_ablation(reference, removed, paths[1])
            plot_count_ablation(
                reference_predictions,
                removed_predictions,
                paths[2],
            )
            plot_fraction_mae_ablation(
                reference_plot_data,
                removed_plot_data,
                paths[3],
            )

            for path in paths:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
