import unittest

import numpy as np
import torch

from ml_ldmx.eval.permutation_aligned import (
    AlignedEventPrediction,
    aligned_event_metrics,
    aligned_event_prediction,
    assign_global_layers,
    confusion_counts,
    energy_weighted_shower_geometry,
    global_layer_z,
)


def _prediction(true_class, predicted_class, energy=None):
    true_class = torch.as_tensor(true_class, dtype=torch.long)
    predicted_class = torch.as_tensor(predicted_class, dtype=torch.long)
    num_hits = true_class.numel()
    num_classes = int(max(true_class.max(), predicted_class.max()).item()) + 1
    logits = torch.full((num_hits, num_classes), -4.0)
    logits[torch.arange(num_hits), predicted_class] = 4.0
    if energy is None:
        energy = torch.ones(num_hits)
    fractions = torch.nn.functional.one_hot(
        true_class,
        num_classes=num_classes,
    ).to(dtype=torch.float32)
    return {
        "event_idx": 7,
        "split_position": 0,
        "true_class": true_class,
        "pred_class": predicted_class,
        "logits": logits,
        "view": {
            "ecal_raw_energy": torch.as_tensor(energy, dtype=torch.float32),
            "ecal_pos": torch.stack(
                [
                    torch.arange(num_hits, dtype=torch.float32),
                    torch.zeros(num_hits),
                    torch.arange(num_hits, dtype=torch.float32) + 10.0,
                ],
                dim=1,
            ),
            "origin_id_fraction_target": fractions,
            "electron_count": torch.tensor(num_classes),
        },
    }


class PermutationAlignedAnalysisTest(unittest.TestCase):
    def test_mapping_direction_recovers_swapped_prediction(self):
        event = aligned_event_prediction(
            _prediction(
                true_class=[0, 0, 1, 1],
                predicted_class=[1, 1, 0, 0],
            ),
            num_classes=2,
        )

        self.assertEqual(event.optimal_mapping.tolist(), [1, 0])
        self.assertEqual(event.aligned_predicted_class.tolist(), [0, 0, 1, 1])
        self.assertTrue(bool(event.aligned_correct.all()))

    def test_global_layer_lookup_preserves_missing_layer(self):
        first = aligned_event_prediction(
            _prediction([0, 1], [0, 1]),
            num_classes=2,
        )
        second = aligned_event_prediction(
            _prediction([0, 0, 1], [0, 0, 1]),
            num_classes=2,
        )
        first.position[:, 2] = np.asarray([10.0, 30.0])
        second.position[:, 2] = np.asarray([10.0, 20.0, 30.0])

        layer_z = global_layer_z([first, second], expected_layers=3)
        assign_global_layers([first, second], layer_z)

        self.assertEqual(layer_z.tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(first.layer.tolist(), [1, 3])
        self.assertEqual(second.layer.tolist(), [1, 2, 3])

    def test_energy_weighted_geometry_drops_zero_fraction_column(self):
        geometry = energy_weighted_shower_geometry(
            position_xy=np.asarray(
                [
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [49.0, 0.0],
                    [51.0, 0.0],
                ]
            ),
            raw_energy=np.ones(4),
            origin_fraction=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            moliere_radius_mm=25.0,
        )

        self.assertEqual(geometry["num_showers"], 2)
        self.assertAlmostEqual(geometry["min_centroid_distance_mm"], 50.0)
        self.assertAlmostEqual(geometry["min_centroid_distance_moliere"], 2.0)
        self.assertAlmostEqual(
            geometry["min_width_normalized_separation"],
            50.0 / np.sqrt(2.0),
        )

    def test_energy_accuracy_reuses_hit_optimal_mapping(self):
        event = aligned_event_prediction(
            _prediction(
                true_class=[0, 0, 0, 1],
                predicted_class=[0, 0, 1, 0],
                energy=[1.0, 1.0, 100.0, 100.0],
            ),
            num_classes=2,
        )
        event.layer = np.asarray([1, 1, 2, 2], dtype=np.int16)

        record = aligned_event_metrics(event)

        self.assertEqual(event.optimal_mapping.tolist(), [0, 1])
        self.assertAlmostEqual(record["aligned_event_accuracy"], 0.5)
        self.assertAlmostEqual(
            record["aligned_energy_weighted_accuracy"],
            2.0 / 202.0,
        )

    def test_depth_confusion_reuses_whole_event_mapping(self):
        event = AlignedEventPrediction(
            event_idx=1,
            split_position=0,
            true_class=np.asarray([0, 0, 1, 1]),
            predicted_class=np.asarray([1, 1, 0, 0]),
            aligned_predicted_class=np.asarray([0, 0, 1, 1]),
            optimal_mapping=np.asarray([1, 0]),
            confidence=np.ones(4),
            normalized_entropy=np.zeros(4),
            probability_margin=np.ones(4),
            raw_energy=np.ones(4),
            position=np.zeros((4, 3)),
            origin_fraction=np.asarray(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
            ),
            layer=np.asarray([1, 2, 3, 4]),
        )

        first_half = confusion_counts(
            [event],
            num_classes=2,
            layer_min=1,
            layer_max=2,
        )
        second_half = confusion_counts(
            [event],
            num_classes=2,
            layer_min=3,
            layer_max=4,
        )

        np.testing.assert_array_equal(first_half, np.asarray([[2, 0], [0, 0]]))
        np.testing.assert_array_equal(second_half, np.asarray([[0, 0], [0, 2]]))


if __name__ == "__main__":
    unittest.main()
