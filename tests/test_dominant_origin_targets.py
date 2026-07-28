import unittest

import torch

from ml_ldmx.datasets.ecal_tpad_loading import (
    apply_target_mode,
    apply_variable_count_target_mode,
)
from ml_ldmx.datasets.tensorize import (
    DOMINANT_ORIGIN_TARGET_RULE,
    LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
    dominant_origin_class_labels,
    origin_energy_fraction_targets,
)


def _event(edeps, origins, *, noise_flags=None):
    num_hits = len(edeps)
    return {
        "x": [float(index) for index in range(num_hits)],
        "hit_id": [100 + index for index in range(num_hits)],
        "noise_flag": (
            list(noise_flags) if noise_flags is not None else [False] * num_hits
        ),
        "edep_contribs": edeps,
        "origin_id_contribs": origins,
    }


class DominantOriginTargetTest(unittest.TestCase):
    def test_summed_origin_energy_can_beat_largest_individual_contribution(self):
        event = _event(
            edeps=[[0.40, 0.35, 0.60]],
            origins=[[1, 1, 2]],
        )

        labels = dominant_origin_class_labels(event, valid_labels=(1, 2))

        self.assertEqual(labels["physical_labels"].tolist(), [1])
        self.assertEqual(labels["class_labels"].tolist(), [0])

    def test_legacy_rule_remains_available_for_old_checkpoint_reproduction(self):
        event = _event(
            edeps=[[0.40, 0.35, 0.60]],
            origins=[[1, 1, 2]],
        )

        labels = dominant_origin_class_labels(
            event,
            valid_labels=(1, 2),
            hard_origin_target_rule=LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
        )

        self.assertEqual(labels["physical_labels"].tolist(), [2])

    def test_target_is_invariant_to_contribution_order(self):
        first = _event(
            edeps=[[0.40, 0.35, 0.60]],
            origins=[[1, 1, 2]],
        )
        reordered = _event(
            edeps=[[0.60, 0.40, 0.35]],
            origins=[[2, 1, 1]],
        )

        first_labels = dominant_origin_class_labels(first, valid_labels=(1, 2))
        reordered_labels = dominant_origin_class_labels(
            reordered,
            valid_labels=(1, 2),
        )
        first_fractions = origin_energy_fraction_targets(
            first,
            first_labels["keep_indices"],
            valid_labels=(1, 2),
        )
        reordered_fractions = origin_energy_fraction_targets(
            reordered,
            reordered_labels["keep_indices"],
            valid_labels=(1, 2),
        )

        self.assertTrue(
            torch.equal(
                first_labels["physical_labels"],
                reordered_labels["physical_labels"],
            )
        )
        self.assertTrue(torch.allclose(first_fractions, reordered_fractions))

    def test_hard_label_agrees_with_largest_origin_fraction(self):
        event = _event(
            edeps=[[0.40, 0.35, 0.60], [0.25, 0.80, 0.10]],
            origins=[[1, 1, 2], [1, 2, 1]],
        )

        labels = dominant_origin_class_labels(event, valid_labels=(1, 2))
        fractions = origin_energy_fraction_targets(
            event,
            labels["keep_indices"],
            valid_labels=(1, 2),
        )
        fraction_labels = torch.tensor((1, 2), dtype=torch.long)[
            fractions.argmax(dim=1)
        ]

        self.assertTrue(torch.equal(labels["physical_labels"], fraction_labels))
        self.assertTrue(
            torch.allclose(
                fractions,
                torch.tensor(
                    [
                        [0.75 / 1.35, 0.60 / 1.35],
                        [0.35 / 1.15, 0.80 / 1.15],
                    ]
                ),
            )
        )

    def test_retained_noise_origin_provenance_uses_summed_energy(self):
        event = _event(
            edeps=[[0.40, 0.35, 0.60], [1.0]],
            origins=[[1, 1, 2], [2]],
            noise_flags=[True, False],
        )

        labels = dominant_origin_class_labels(
            event,
            valid_labels=(1, 2),
            filter_noise=False,
            supervise_noise=True,
        )

        self.assertEqual(labels["physical_labels"].tolist(), [0, 2])
        self.assertEqual(labels["origin_id_labels"].tolist(), [1, 2])
        self.assertEqual(labels["is_noise_target"].tolist(), [True, False])

    def test_retained_noise_without_contributions_keeps_unknown_provenance(self):
        event = _event(
            edeps=[[], [1.0]],
            origins=[[], [2]],
            noise_flags=[True, False],
        )

        labels = dominant_origin_class_labels(
            event,
            valid_labels=(1, 2),
            filter_noise=False,
            supervise_noise=True,
        )

        self.assertEqual(labels["physical_labels"].tolist(), [0, 2])
        self.assertEqual(labels["origin_id_labels"].tolist(), [-1, 2])

    def test_mismatched_contribution_arrays_are_rejected(self):
        event = _event(
            edeps=[[0.6, 0.4]],
            origins=[[1]],
        )

        with self.assertRaisesRegex(
            ValueError,
            "2 edep contributions but 1 origin contributions",
        ):
            dominant_origin_class_labels(event, valid_labels=(1, 2))

        with self.assertRaisesRegex(
            ValueError,
            "2 edep contributions but 1 origin contributions",
        ):
            origin_energy_fraction_targets(
                event,
                keep_indices=[0],
                valid_labels=(1, 2),
            )

    def test_dominant_origin_outside_valid_labels_is_rejected_after_summing(self):
        event = _event(
            edeps=[[0.40, 0.35, 0.60]],
            origins=[[7, 7, 1]],
        )

        with self.assertRaisesRegex(ValueError, "dominant origin label 7"):
            dominant_origin_class_labels(event, valid_labels=(1, 2))


class CachedDominantOriginMigrationTest(unittest.TestCase):
    def _cached_event(self):
        fractions = torch.tensor(
            [
                [0.75 / 1.35, 0.60 / 1.35],
                [0.70, 0.30],
                [0.20, 0.80],
            ],
            dtype=torch.float32,
        )
        return {
            # These labels represent the previous largest-individual-contribution
            # rule and deliberately disagree with the summed fraction targets.
            "physical_y": torch.tensor([2, 2, 1], dtype=torch.long),
            "origin_id_y": torch.tensor([2, 2, 1], dtype=torch.long),
            "y": torch.tensor([1, 1, 0], dtype=torch.long),
            "ecal_pos": torch.tensor(
                [
                    [0.0, -2.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "fraction_target": fractions,
            "target_label_order": [1, 2],
            "event_idx": torch.tensor(17),
        }

    def test_new_rule_relabels_cached_event_before_canonicalization(self):
        event = self._cached_event()

        apply_variable_count_target_mode(
            event,
            valid_labels=(1, 2),
            target_mode="canonical-y",
            max_electrons=2,
            hard_origin_target_rule=DOMINANT_ORIGIN_TARGET_RULE,
        )

        self.assertEqual(event["origin_id_y"].tolist(), [1, 1, 2])
        self.assertEqual(event["target_label_order"], [1, 2])
        self.assertEqual(event["y"].tolist(), [0, 0, 1])

    def test_legacy_rule_preserves_cached_labels(self):
        event = self._cached_event()

        apply_variable_count_target_mode(
            event,
            valid_labels=(1, 2),
            target_mode="canonical-y",
            max_electrons=2,
            hard_origin_target_rule=LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
        )

        self.assertEqual(event["origin_id_y"].tolist(), [2, 2, 1])
        self.assertEqual(event["target_label_order"], [2, 1])
        self.assertEqual(event["y"].tolist(), [0, 0, 1])

    def test_cached_fraction_label_order_is_used_for_physical_targets(self):
        event = {
            "physical_y": torch.tensor([2, 1], dtype=torch.long),
            "origin_id_y": torch.tensor([2, 1], dtype=torch.long),
            "y": torch.tensor([1, 0], dtype=torch.long),
            "ecal_pos": torch.zeros((2, 3), dtype=torch.float32),
            # Column zero is origin 2 and column one is origin 1.
            "fraction_target": torch.tensor(
                [[0.20, 0.80], [0.90, 0.10]],
                dtype=torch.float32,
            ),
            "target_label_order": [2, 1],
            "event_idx": torch.tensor(18),
        }

        apply_target_mode(
            event,
            valid_labels=(1, 2),
            target_mode="physical-origin",
            hard_origin_target_rule=DOMINANT_ORIGIN_TARGET_RULE,
        )

        self.assertEqual(event["physical_y"].tolist(), [1, 2])
        self.assertEqual(event["origin_id_y"].tolist(), [1, 2])
        self.assertEqual(event["y"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
