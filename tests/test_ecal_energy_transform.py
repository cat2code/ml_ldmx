import math
import unittest

import torch

from ml_ldmx.datasets.ecal_tpad_loading import ecal_tpad_event_to_tensors
from ml_ldmx.datasets.model_views import ecal_tpad_transformer_view, ecal_transformer_view
from ml_ldmx.datasets.tensorize import tensorize_ecal_event, tensorize_ecal_with_triggerpad_context


def _event():
    return {
        "x": [1.0, 2.0],
        "y": [3.0, 4.0],
        "z": [5.0, 6.0],
        "energy": [0.0, 9.0],
        "hit_id": [11, 12],
        "noise_flag": [False, False],
        "edep_contribs": [[2.0], [3.0]],
        "origin_id_contribs": [[1], [2]],
        "trigger_pad_tracks": {
            "centroid": [7.0],
            "pe": [8.0],
        },
    }


class FeatureTransformTest(unittest.TestCase):
    def test_tensorize_ecal_event_log1p_transforms_reconstructed_energy_only(self):
        raw_x, _pos = tensorize_ecal_event(_event())
        log_x, _pos = tensorize_ecal_event(_event(), ecal_energy_transform="log1p")

        self.assertTrue(torch.allclose(raw_x[:, 3], torch.tensor([0.0, 9.0])))
        self.assertTrue(torch.allclose(log_x[:, 3], torch.tensor([0.0, math.log1p(9.0)])))
        self.assertTrue(torch.allclose(raw_x[:, :3], log_x[:, :3]))

    def test_combined_tensor_places_log1p_inputs_in_expected_columns(self):
        tensors = tensorize_ecal_with_triggerpad_context(
            _event(),
            valid_labels=(1, 2),
            ecal_energy_transform="log1p",
            tpad_pe_transform="log1p",
        )

        self.assertTrue(torch.allclose(tensors["x"][:2, 5], torch.tensor([0.0, math.log1p(9.0)])))
        self.assertEqual(float(tensors["x"][2, 5]), 0.0)
        self.assertEqual(float(tensors["x"][2, 6]), 7.0)
        self.assertTrue(torch.allclose(tensors["x"][2, 7], torch.tensor(math.log1p(8.0))))
        self.assertTrue(torch.allclose(tensors["tpad"][0], torch.tensor([7.0, math.log1p(8.0)])))
        self.assertTrue(torch.allclose(tensors["tpad_raw_pe"], torch.tensor([8.0])))
        self.assertTrue(
            torch.allclose(
                tensors["ecal_input_energy"],
                torch.tensor([0.0, math.log1p(9.0)]),
            )
        )
        self.assertTrue(torch.allclose(tensors["ecal_raw_energy"], torch.tensor([0.0, 9.0])))

    def test_fraction_targets_remain_in_truth_energy_units(self):
        tensors = ecal_tpad_event_to_tensors(
            _event(),
            event_idx=0,
            valid_labels=(1, 2),
            ecal_energy_transform="log1p",
            tpad_pe_transform="log1p",
        )

        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(torch.allclose(tensors["fraction_target"], expected))

    def test_raw_tpad_pe_is_available_only_to_combined_views(self):
        tensors = tensorize_ecal_with_triggerpad_context(
            _event(),
            valid_labels=(1, 2),
            tpad_pe_transform="log1p",
        )

        combined_view = ecal_tpad_transformer_view(tensors)
        ecal_view = ecal_transformer_view(tensors)

        self.assertTrue(torch.equal(combined_view["tpad_raw_pe"], torch.tensor([8.0])))
        self.assertNotIn("tpad_raw_pe", ecal_view)

    def test_unknown_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            tensorize_ecal_event(_event(), ecal_energy_transform="sqrt")

        with self.assertRaises(ValueError):
            tensorize_ecal_with_triggerpad_context(
                _event(),
                valid_labels=(1, 2),
                tpad_pe_transform="sqrt",
            )


if __name__ == "__main__":
    unittest.main()
