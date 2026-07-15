import unittest

import torch

from ml_ldmx.models import ECalTpadTrackSeededTransformer
from ml_ldmx.train.hit_classifier_batching import (
    collate_transformer_hit_classifier_batch,
    hit_classifier_batch_kind,
)


def _event(num_ecal, tpad_values, feature_dim=8):
    ecal = torch.randn(num_ecal, feature_dim)
    ecal[:, 0] = 1.0
    ecal[:, 1] = 0.0
    tpad = torch.zeros(len(tpad_values), feature_dim)
    if tpad_values:
        tpad[:, 0] = 0.0
        tpad[:, 1] = 1.0
        tpad[:, 6] = torch.tensor(tpad_values, dtype=torch.float32)
        tpad[:, 7] = 1.0
    x = torch.cat([ecal, tpad], dim=0)
    return {
        "x": x,
        "ecal_mask": torch.tensor(
            [True] * num_ecal + [False] * len(tpad_values),
            dtype=torch.bool,
        ),
        "y": torch.arange(num_ecal, dtype=torch.long) % 3,
    }


class TrackSeededTransformerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = ECalTpadTrackSeededTransformer(
            in_dim=8,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            out_dim=3,
        )

    def test_forward_supports_missing_and_extra_tpad_tokens(self):
        self.model.eval()
        for tpad_values in ([], [-2.0], [-2.0, 0.0, 2.0], [-3.0, -1.0, 1.0, 3.0]):
            with self.subTest(num_tpad=len(tpad_values)):
                event = _event(5, tpad_values)
                logits = self.model(event["x"])
                self.assertEqual(tuple(logits.shape), (len(event["x"]), 3))
                self.assertTrue(bool(torch.isfinite(logits).all().item()))

    def test_batched_forward_matches_single_event_forward(self):
        self.model.eval()
        events = [_event(5, [-1.0, 1.0]), _event(3, [])]
        batch = collate_transformer_hit_classifier_batch(events)

        with torch.no_grad():
            expected = self.model(events[0]["x"])
            actual = self.model(batch.x, key_padding_mask=~batch.valid_mask)

        torch.testing.assert_close(expected, actual[0, : len(events[0]["x"])], atol=1e-6, rtol=1e-6)

    def test_backward_reaches_slots_and_attention(self):
        self.model.train()
        event = _event(6, [-1.0, 1.0])
        logits = self.model(event["x"])[event["ecal_mask"]]
        loss = torch.nn.functional.cross_entropy(logits, event["y"])
        loss.backward()

        self.assertIsNotNone(self.model.canonical_slots.grad)
        self.assertIsNotNone(self.model.track_attention.in_proj_weight.grad)
        self.assertIsNotNone(self.model.shower_attention.in_proj_weight.grad)
        self.assertEqual(hit_classifier_batch_kind(self.model), "padded")


if __name__ == "__main__":
    unittest.main()
