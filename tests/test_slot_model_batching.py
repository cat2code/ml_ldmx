from types import SimpleNamespace
import logging
import unittest

import torch

from ml_ldmx.eval.ecal_tpad_slot_model import evaluate
from ml_ldmx.models import ECalTpadSlotModel
from ml_ldmx.train.ecal_tpad_slot_batching import (
    IGNORE_INDEX,
    collate_ecal_tpad_slot_batch,
)
from ml_ldmx.train.ecal_tpad_slot_model import (
    count_cross_entropy_per_event,
    compute_batch_losses,
    compute_event_losses,
    empty_slot_metric_totals,
    finalize_slot_metrics,
    train_one_epoch,
    update_slot_metric_totals,
)


def _event(x, ecal_mask, physical_y, fraction_target, electron_count, event_idx):
    return {
        "x": torch.as_tensor(x, dtype=torch.float32),
        "ecal_mask": torch.as_tensor(ecal_mask, dtype=torch.bool),
        "physical_y": torch.as_tensor(physical_y, dtype=torch.long),
        "fraction_target": torch.as_tensor(fraction_target, dtype=torch.float32),
        "electron_count": int(electron_count),
        "event_idx": int(event_idx),
    }


def _events():
    return [
        _event(
            [
                [1.0, 0.0, 0.1, 0.2],
                [0.0, 1.0, 0.3, 0.4],
                [1.0, 0.0, 0.5, 0.6],
                [0.0, 1.0, 0.7, 0.8],
                [1.0, 0.0, 0.9, 1.0],
            ],
            [True, False, True, False, True],
            [1, 0, 2],
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.8, 0.0],
            ],
            electron_count=2,
            event_idx=101,
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
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.9],
            ],
            electron_count=3,
            event_idx=202,
        ),
    ]


def _args(batch_size=2):
    return SimpleNamespace(
        lambda_origin=1.0,
        lambda_fraction=0.7,
        lambda_slot=0.5,
        lambda_count=1.3,
        origin_class_weights=[0.8, 1.4, 0.9, 1.1],
        count_class_weights=[0.5, 1.7, 0.6, 1.2],
        batch_size=batch_size,
        max_electrons=3,
    )


def _model():
    return ECalTpadSlotModel(
        in_dim=4,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        max_electrons=3,
        dropout=0.0,
        use_type_embedding=True,
    )


class SlotModelBatchingTest(unittest.TestCase):
    def test_training_uses_dataset_source_balanced_batches(self):
        class BalancedEvents:
            def __init__(self, events):
                self.events = events
                self.request = None

            def __getitem__(self, index):
                return self.events[index]

            def balanced_batches_for_access(self, indices, batch_size, seed=None):
                self.request = (list(indices), int(batch_size), int(seed))
                return [[0, 1]]

        torch.manual_seed(9)
        events = BalancedEvents(_events())
        model = _model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = _args(batch_size=2)
        args.seed = 7
        args.epochs = 1
        args.no_progress = True
        args.grad_clip = 1.0

        metrics = train_one_epoch(
            model,
            events,
            [0, 1],
            optimizer,
            args,
            torch.device("cpu"),
            epoch=0,
            logger=logging.getLogger("slot-balanced-batch-test"),
        )

        self.assertEqual(events.request, ([0, 1], 2, 7))
        self.assertEqual(metrics["train_num_events"], 2)

    def test_count_class_weights_change_each_events_loss_at_fixed_scale(self):
        logits = torch.zeros((2, 4), dtype=torch.float32)
        target = torch.tensor([2, 3], dtype=torch.long)
        args = SimpleNamespace(count_class_weights=[0.0, 0.0, 0.5, 1.5])

        losses = count_cross_entropy_per_event(logits, target, args)
        base_loss = torch.log(torch.tensor(4.0))

        self.assertTrue(torch.allclose(losses, base_loss * torch.tensor([0.5, 1.5])))

    def test_collation_pads_inputs_and_aligns_all_targets(self):
        batch = collate_ecal_tpad_slot_batch(_events(), max_electrons=3)

        self.assertEqual(tuple(batch.x.shape), (2, 5, 4))
        self.assertTrue(
            torch.equal(
                batch.valid_mask,
                torch.tensor(
                    [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
                    dtype=torch.bool,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.ecal_mask,
                torch.tensor(
                    [[1, 0, 1, 0, 1], [1, 1, 0, 0, 0]],
                    dtype=torch.bool,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.origin_target,
                torch.tensor(
                    [
                        [1, IGNORE_INDEX, 0, IGNORE_INDEX, 2],
                        [1, 3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(batch.slot_target[0], torch.tensor([1.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(batch.slot_target[1], torch.tensor([1.0, 1.0, 1.0])))
        self.assertTrue(torch.equal(batch.count_target, torch.tensor([2, 3])))

    def test_masked_batched_forward_matches_individual_events(self):
        torch.manual_seed(11)
        model = _model().eval()
        events = _events()
        batch = collate_ecal_tpad_slot_batch(events, max_electrons=3)

        with torch.no_grad():
            batched = model(
                batch.x,
                ecal_mask=batch.ecal_mask,
                key_padding_mask=~batch.valid_mask,
            )
            singles = [
                model(event["x"], ecal_mask=event["ecal_mask"])
                for event in events
            ]

        for row, (event, single) in enumerate(zip(events, singles)):
            num_tokens = int(event["x"].shape[0])
            for key in ("origin_logits", "fraction_logits", "fraction_pred"):
                self.assertTrue(
                    torch.allclose(batched[key][row, :num_tokens], single[key], atol=1e-6),
                    key,
                )
            for key in ("slot_valid_logits", "count_logits", "signal_logit"):
                self.assertTrue(torch.allclose(batched[key][row], single[key], atol=1e-6), key)

    def test_batch_loss_and_metrics_match_serial_event_computation(self):
        torch.manual_seed(13)
        model = _model().eval()
        events = _events()
        args = _args()

        batched = compute_batch_losses(model, events, torch.device("cpu"), args)
        singles = [
            compute_event_losses(model, event, torch.device("cpu"), args)
            for event in events
        ]

        for key in ("total_loss", "origin_loss", "fraction_loss", "slot_loss", "count_loss"):
            expected = torch.stack([loss[key] for loss in singles]).mean()
            self.assertTrue(torch.allclose(batched[key], expected, atol=1e-6), key)
        expected_mse = sum(loss["fraction_mse"] * loss["num_hits"] for loss in singles) / sum(
            loss["num_hits"] for loss in singles
        )
        expected_mae = sum(loss["fraction_mae"] * loss["num_hits"] for loss in singles) / sum(
            loss["num_hits"] for loss in singles
        )
        self.assertTrue(torch.allclose(batched["fraction_mse"], expected_mse, atol=1e-6))
        self.assertTrue(torch.allclose(batched["fraction_mae"], expected_mae, atol=1e-6))

        serial_totals = empty_slot_metric_totals(4, 4)
        batch_totals = empty_slot_metric_totals(4, 4)
        for loss in singles:
            update_slot_metric_totals(serial_totals, loss)
        update_slot_metric_totals(batch_totals, batched)
        self.assertEqual(serial_totals["events"], batch_totals["events"])
        self.assertEqual(serial_totals["hits"], batch_totals["hits"])
        self.assertTrue(torch.equal(serial_totals["hit_confusion"], batch_totals["hit_confusion"]))
        self.assertTrue(
            torch.equal(serial_totals["count_confusion"], batch_totals["count_confusion"])
        )
        serial_metrics = finalize_slot_metrics(serial_totals)
        batch_metrics = finalize_slot_metrics(batch_totals)
        for key in serial_metrics:
            if isinstance(serial_metrics[key], float):
                self.assertAlmostEqual(serial_metrics[key], batch_metrics[key], places=6, msg=key)
            else:
                self.assertEqual(serial_metrics[key], batch_metrics[key], key)

    def test_batched_objective_backpropagates_through_all_trained_heads(self):
        torch.manual_seed(17)
        model = _model().train()
        losses = compute_batch_losses(model, _events(), torch.device("cpu"), _args())
        losses["total_loss"].backward()

        for module_name in (
            "input_proj",
            "encoder",
            "origin_head",
            "fraction_head",
            "event_summary",
            "slot_valid_head",
            "count_head",
        ):
            module = getattr(model, module_name)
            self.assertTrue(
                any(parameter.grad is not None for parameter in module.parameters()),
                module_name,
            )

    def test_batched_gradients_match_serial_event_accumulation(self):
        torch.manual_seed(18)
        serial_model = _model().eval()
        batched_model = _model().eval()
        batched_model.load_state_dict(serial_model.state_dict())
        events = _events()
        args = _args()

        serial_loss = torch.stack(
            [
                compute_event_losses(serial_model, event, torch.device("cpu"), args)[
                    "total_loss"
                ]
                for event in events
            ]
        ).mean()
        serial_loss.backward()
        compute_batch_losses(
            batched_model,
            events,
            torch.device("cpu"),
            args,
        )["total_loss"].backward()

        serial_parameters = dict(serial_model.named_parameters())
        for name, batched_parameter in batched_model.named_parameters():
            serial_gradient = serial_parameters[name].grad
            batched_gradient = batched_parameter.grad
            if serial_gradient is None or batched_gradient is None:
                self.assertIsNone(serial_gradient, name)
                self.assertIsNone(batched_gradient, name)
            else:
                self.assertTrue(
                    torch.allclose(serial_gradient, batched_gradient, atol=2e-6, rtol=1e-5),
                    name,
                )

    def test_evaluation_uses_true_batches_and_keeps_prediction_records(self):
        torch.manual_seed(19)
        model = _model()
        metrics, predictions = evaluate(
            model,
            _events(),
            [0, 1],
            _args(batch_size=2),
            torch.device("cpu"),
            "test",
            collect_predictions=True,
        )

        self.assertEqual(metrics["test_num_events"], 2)
        self.assertEqual(metrics["test_num_hits"], 5)
        self.assertEqual([row["event_id"] for row in predictions], [101, 202])


if __name__ == "__main__":
    unittest.main()
