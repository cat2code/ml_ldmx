from types import SimpleNamespace
import logging
import unittest

import torch

from ml_ldmx.eval.contributor_set_postprocessing import (
    fraction_targets_to_support,
    postprocess_contributor_set_outputs,
    support_bit_table,
)
from ml_ldmx.eval.ecal_tpad_contributor_set_slot_model import evaluate
from ml_ldmx.models import ECalTpadContributorSetSlotModel, ECalTpadSlotModel
from ml_ldmx.train.ecal_tpad_contributor_set_slot_model import (
    compute_batch_losses,
    train_one_epoch,
)


def _event(x, ecal_mask, physical_y, fraction_target, electron_count, event_idx):
    num_ecal = sum(ecal_mask)
    return {
        "x": torch.as_tensor(x, dtype=torch.float32),
        "ecal_mask": torch.as_tensor(ecal_mask, dtype=torch.bool),
        "physical_y": torch.as_tensor(physical_y, dtype=torch.long),
        "fraction_target": torch.as_tensor(fraction_target, dtype=torch.float32),
        "electron_count": int(electron_count),
        "event_idx": int(event_idx),
        "ecal_pos": torch.arange(num_ecal * 3, dtype=torch.float32).reshape(num_ecal, 3),
    }


def _events():
    return [
        _event(
            [
                [1.0, 0.0, 0.1, 0.2],
                [0.0, 1.0, 0.3, 0.4],
                [1.0, 0.0, 0.5, 0.6],
                [1.0, 0.0, 0.7, 0.8],
            ],
            [True, False, True, True],
            [1, 2, 0],
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.25, 0.75, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            2,
            101,
        ),
        _event(
            [
                [1.0, 0.0, -0.2, 0.1],
                [1.0, 0.0, 0.4, -0.3],
                [0.0, 1.0, 0.2, 0.5],
            ],
            [True, True, False],
            [1, 3],
            [
                [0.0, 0.4, 0.0, 0.6],
                [0.0, 0.0, 0.2, 0.8],
            ],
            3,
            202,
        ),
    ]


def _model():
    return ECalTpadContributorSetSlotModel(
        in_dim=4,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        max_electrons=3,
        min_electrons=2,
        dropout=0.0,
    )


def _args(batch_size=2):
    return SimpleNamespace(
        contribution_epsilon=0.0,
        support_class_weights=[1.0] * 8,
        lambda_support=1.0,
        lambda_fraction=0.7,
        lambda_slot=1.0,
        batch_size=batch_size,
        seed=7,
        epochs=1,
        no_progress=True,
        grad_clip=1.0,
    )


class ContributorSetPostprocessingTest(unittest.TestCase):
    def test_support_encoding_covers_noise_pure_and_mixed_hits(self):
        fractions = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.2, 0.8, 0.0],
                [0.0, 0.1, 0.2, 0.7],
            ]
        )
        self.assertTrue(
            torch.equal(fraction_targets_to_support(fractions), torch.tensor([0, 1, 3, 7]))
        )
        self.assertTrue(
            torch.equal(
                support_bit_table(3),
                torch.tensor(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1],
                    ],
                    dtype=torch.bool,
                ),
            )
        )

    def test_slot_count_masks_illegal_support_and_gates_fractions(self):
        support_logits = torch.full((1, 2, 8), -10.0)
        support_logits[0, 0, 4] = 20.0  # e3: highest raw score, illegal for predicted 2e.
        support_logits[0, 0, 3] = 10.0  # e1+e2: highest legal score.
        support_logits[0, 1, 0] = 10.0  # explicit noise.
        fraction_logits = torch.tensor(
            [[[0.0, 1.0, 2.0, 50.0], [-5.0, 10.0, 10.0, 10.0]]]
        )
        outputs = {
            "support_logits": support_logits,
            "fraction_logits": fraction_logits,
            "slot_valid_logits": torch.tensor([[10.0, 10.0, -10.0]]),
        }
        result = postprocess_contributor_set_outputs(outputs, min_electrons=2)

        self.assertEqual(result["predicted_count"].tolist(), [2])
        self.assertEqual(result["support_prediction"].tolist(), [[3, 0]])
        self.assertTrue(result["mixed_prediction"][0, 0])
        self.assertFalse(result["mixed_prediction"][0, 1])
        self.assertEqual(float(result["fraction_prediction"][0, 0, 0]), 0.0)
        self.assertEqual(float(result["fraction_prediction"][0, 0, 3]), 0.0)
        self.assertTrue(
            torch.allclose(result["fraction_prediction"].sum(dim=-1), torch.ones((1, 2)))
        )
        self.assertTrue(torch.equal(result["fraction_prediction"][0, 1], torch.tensor([1.0, 0, 0, 0])))

    def test_mixed_probability_is_learned_support_mass_not_a_fraction_threshold(self):
        support_logits = torch.log(
            torch.tensor([[[0.05, 0.10, 0.10, 0.20, 0.10, 0.15, 0.10, 0.20]]])
        )
        result = postprocess_contributor_set_outputs(
            {
                "support_logits": support_logits,
                "fraction_logits": torch.zeros((1, 1, 4)),
                "slot_valid_logits": torch.full((1, 3), 10.0),
            },
            min_electrons=2,
        )
        self.assertAlmostEqual(float(result["mixed_probability"][0, 0]), 0.65, places=5)


class ContributorSetSlotModelTest(unittest.TestCase):
    def test_new_model_is_parallel_to_and_distinct_from_legacy_slot_model(self):
        new_model = _model()
        old_model = ECalTpadSlotModel(
            in_dim=4,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            max_electrons=3,
            dropout=0.0,
        )
        self.assertTrue(hasattr(new_model, "support_head"))
        self.assertFalse(hasattr(new_model, "count_head"))
        self.assertFalse(hasattr(new_model, "origin_head"))
        self.assertTrue(hasattr(old_model, "count_head"))
        self.assertTrue(hasattr(old_model, "origin_head"))

    def test_masked_batch_forward_matches_individual_events(self):
        torch.manual_seed(11)
        model = _model().eval()
        events = _events()
        max_tokens = max(event["x"].shape[0] for event in events)
        x = torch.zeros((2, max_tokens, 4))
        padding = torch.ones((2, max_tokens), dtype=torch.bool)
        for row, event in enumerate(events):
            length = event["x"].shape[0]
            x[row, :length] = event["x"]
            padding[row, :length] = False
        with torch.no_grad():
            batched = model(x, key_padding_mask=padding)
            singles = [model(event["x"]) for event in events]
        for row, (event, single) in enumerate(zip(events, singles)):
            length = event["x"].shape[0]
            for key in ("support_logits", "fraction_logits", "raw_fraction_pred"):
                self.assertTrue(torch.allclose(batched[key][row, :length], single[key], atol=1e-6), key)
            self.assertTrue(
                torch.allclose(batched["slot_valid_logits"][row], single["slot_valid_logits"], atol=1e-6)
            )

    def test_objective_reaches_every_trained_head_and_shared_backbone(self):
        torch.manual_seed(13)
        model = _model().train()
        losses = compute_batch_losses(model, _events(), torch.device("cpu"), _args())
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        for name in (
            "input_proj",
            "encoder",
            "support_head",
            "fraction_head",
            "event_summary",
            "slot_valid_head",
        ):
            self.assertTrue(
                any(parameter.grad is not None for parameter in getattr(model, name).parameters()),
                name,
            )

    def test_training_and_evaluation_use_true_event_batching(self):
        class BalancedEvents:
            def __init__(self, events):
                self.events = events
                self.request = None

            def __getitem__(self, index):
                return self.events[index]

            def balanced_batches_for_access(self, indices, batch_size, seed=None):
                self.request = (list(indices), int(batch_size), int(seed))
                return [[0, 1]]

            def order_indices_for_access(self, indices, seed=None):
                return list(indices)

        events = BalancedEvents(_events())
        model = _model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_one_epoch(
            model,
            events,
            [0, 1],
            optimizer,
            _args(),
            torch.device("cpu"),
            epoch=0,
            logger=logging.getLogger("contributor-set-test"),
        )
        self.assertEqual(events.request, ([0, 1], 2, 7))
        self.assertEqual(metrics["train_num_events"], 2)
        self.assertIn("train_support_accuracy", metrics)
        self.assertIn("train_mixed_f1", metrics)

        evaluated, predictions, plot_data = evaluate(
            model,
            events,
            [0, 1],
            _args(),
            torch.device("cpu"),
            "test",
            collect_predictions=True,
            max_plot_hits=3,
        )
        self.assertEqual(evaluated["test_num_events"], 2)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(plot_data["support_target"].shape[0], 3)
        self.assertEqual(len(evaluated["test_support_confusion"]), 8)


if __name__ == "__main__":
    unittest.main()
