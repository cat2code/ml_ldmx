import unittest

import torch

from ml_ldmx.train.hit_classifier_baseline import (
    empty_metric_totals as empty_baseline_totals,
    finalize_metrics as finalize_baseline_metrics,
    update_metric_totals as update_baseline_totals,
)
from ml_ldmx.train.metrics import (
    classification_metrics_from_confusion,
    confusion_components_from_confusion,
    confusion_matrix_from_class_indices,
    confusion_matrix_from_labels,
    empty_metric_totals,
    finalize_metrics,
    update_metric_totals,
)
from ml_ldmx.train.ecal_tpad_slot_model import (
    empty_slot_metric_totals,
    finalize_slot_metrics,
    update_slot_metric_totals,
)
from ml_ldmx.viz.event_level import _confusion_from_labels, plot_event_count_confusion_matrix
from ml_ldmx.viz.training import plot_confusion_matrix


def _old_baseline_update(totals, losses):
    num_hits = int(losses["num_hits"])
    totals["loss_sum"] += float(losses["total_loss"].detach().cpu().item()) * num_hits
    pred = losses["pred_class"].detach().cpu()
    true = losses["true_class"].detach().cpu()
    totals["correct"] += int((pred == true).sum().item())
    totals["hits"] += num_hits
    for true_idx, pred_idx in zip(true.tolist(), pred.tolist()):
        totals["confusion"][int(true_idx), int(pred_idx)] += 1


def _old_fraction_update(totals, losses):
    num_hits = int(losses["num_hits"])
    totals["loss_sum"] += float(losses["total_loss"].detach().cpu().item()) * num_hits
    totals["origin_loss_sum"] += float(losses["origin_loss"].detach().cpu().item()) * num_hits
    totals["fraction_loss_sum"] += float(losses["fraction_loss"].detach().cpu().item()) * num_hits
    totals["fraction_mse_sum"] += float(losses["fraction_mse"].detach().cpu().item()) * num_hits
    totals["fraction_mae_sum"] += float(losses["fraction_mae"].detach().cpu().item()) * num_hits
    pred = losses["pred_class"].detach().cpu()
    true = losses["true_class"].detach().cpu()
    totals["correct"] += int((pred == true).sum().item())
    totals["hits"] += num_hits
    for true_idx, pred_idx in zip(true.tolist(), pred.tolist()):
        totals["confusion"][int(true_idx), int(pred_idx)] += 1


def _old_slot_update(totals, losses):
    num_hits = int(losses["num_hits"])
    totals["loss_sum"] += float(losses["total_loss"].detach().cpu().item())
    totals["origin_loss_sum"] += float(losses["origin_loss"].detach().cpu().item())
    totals["fraction_loss_sum"] += float(losses["fraction_loss"].detach().cpu().item())
    totals["slot_loss_sum"] += float(losses["slot_loss"].detach().cpu().item())
    totals["count_loss_sum"] += float(losses["count_loss"].detach().cpu().item())
    totals["fraction_mse_sum"] += float(losses["fraction_mse"].detach().cpu().item()) * num_hits
    totals["fraction_mae_sum"] += float(losses["fraction_mae"].detach().cpu().item()) * num_hits

    pred = losses["pred_class"].detach().cpu()
    true = losses["true_class"].detach().cpu()
    totals["correct_hits"] += int((pred == true).sum().item())
    totals["hits"] += num_hits
    for true_idx, pred_idx in zip(true.tolist(), pred.tolist()):
        totals["hit_confusion"][int(true_idx), int(pred_idx)] += 1

    slot_pred = losses["slot_pred"].detach().cpu()
    slot_true = losses["slot_target"].detach().cpu().to(dtype=torch.bool)
    totals["slot_correct"] += int((slot_pred == slot_true).sum().item())
    totals["slot_total"] += int(slot_true.numel())
    totals["slot_exact_correct"] += int(bool((slot_pred == slot_true).all().item()))

    count_true = int(losses["count_target"].detach().cpu().item())
    count_pred = int(losses["count_pred"].detach().cpu().item())
    slot_count_pred = int(losses["slot_count_pred"].detach().cpu().item())
    totals["count_correct"] += int(count_pred == count_true)
    totals["slot_count_correct"] += int(slot_count_pred == count_true)
    totals["count_total_by_true"][count_true] = totals["count_total_by_true"].get(count_true, 0) + 1
    totals["count_correct_by_true"][count_true] = totals["count_correct_by_true"].get(count_true, 0) + int(
        count_pred == count_true
    )
    totals["count_confusion"][count_true, count_pred] += 1
    totals["events"] += 1


class VectorizedMetricsTest(unittest.TestCase):
    def test_confusion_matrix_from_class_indices_matches_manual_counts(self):
        true = torch.tensor([0, 1, 2, 1, 2, 2, 0])
        pred = torch.tensor([0, 2, 2, 1, 0, 2, 1])

        expected = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 2],
            ]
        )

        actual = confusion_matrix_from_class_indices(true, pred, num_classes=3)
        self.assertTrue(torch.equal(actual, expected))

        tp, fp, tn, fn = confusion_components_from_confusion(actual)
        self.assertTrue(torch.equal(tp.to(dtype=torch.long), torch.tensor([1, 1, 2])))
        self.assertTrue(torch.equal(fp.to(dtype=torch.long), torch.tensor([1, 1, 1])))
        self.assertTrue(torch.equal(fn.to(dtype=torch.long), torch.tensor([1, 1, 1])))
        self.assertTrue(torch.equal(tn.to(dtype=torch.long), torch.tensor([4, 4, 3])))

    def test_confusion_matrix_from_labels_respects_label_order_and_ignores_unknowns(self):
        true = [20, 10, 30, 99, 20]
        pred = [20, 30, 30, 10, 99]
        labels = [10, 20, 30]

        expected = torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        actual = confusion_matrix_from_labels(true, pred, labels)
        self.assertTrue(torch.equal(actual.cpu(), expected))
        self.assertTrue(torch.equal(_confusion_from_labels(true, pred, labels), expected))

    def test_baseline_update_matches_old_loop_update(self):
        losses = {
            "total_loss": torch.tensor(1.25),
            "pred_class": torch.tensor([0, 2, 2, 1, 0, 1]),
            "true_class": torch.tensor([0, 1, 2, 1, 2, 1]),
            "num_hits": 6,
        }
        old_totals = empty_baseline_totals(num_classes=3)
        new_totals = empty_baseline_totals(num_classes=3)

        _old_baseline_update(old_totals, losses)
        update_baseline_totals(new_totals, losses)

        self.assertEqual(old_totals["loss_sum"], new_totals["loss_sum"])
        self.assertEqual(old_totals["correct"], new_totals["correct"])
        self.assertEqual(old_totals["hits"], new_totals["hits"])
        self.assertTrue(torch.equal(old_totals["confusion"], new_totals["confusion"]))
        self.assertEqual(
            finalize_baseline_metrics(old_totals),
            finalize_baseline_metrics(new_totals),
        )

    def test_fraction_update_matches_old_loop_update(self):
        losses = {
            "total_loss": torch.tensor(2.0),
            "origin_loss": torch.tensor(1.5),
            "fraction_loss": torch.tensor(0.5),
            "fraction_mse": torch.tensor(0.25),
            "fraction_mae": torch.tensor(0.1),
            "pred_class": torch.tensor([0, 2, 2, 1, 0, 1]),
            "true_class": torch.tensor([0, 1, 2, 1, 2, 1]),
            "num_hits": 6,
        }
        old_totals = empty_metric_totals(num_classes=3)
        new_totals = empty_metric_totals(num_classes=3)

        _old_fraction_update(old_totals, losses)
        update_metric_totals(new_totals, losses)

        self.assertEqual(old_totals["loss_sum"], new_totals["loss_sum"])
        self.assertEqual(old_totals["origin_loss_sum"], new_totals["origin_loss_sum"])
        self.assertEqual(old_totals["fraction_loss_sum"], new_totals["fraction_loss_sum"])
        self.assertEqual(old_totals["fraction_mse_sum"], new_totals["fraction_mse_sum"])
        self.assertEqual(old_totals["fraction_mae_sum"], new_totals["fraction_mae_sum"])
        self.assertEqual(old_totals["correct"], new_totals["correct"])
        self.assertTrue(torch.equal(old_totals["confusion"], new_totals["confusion"]))
        self.assertEqual(finalize_metrics(old_totals), finalize_metrics(new_totals))
        self.assertEqual(
            classification_metrics_from_confusion(old_totals["confusion"]),
            classification_metrics_from_confusion(new_totals["confusion"]),
        )

    def test_slot_update_matches_old_loop_update(self):
        losses = {
            "total_loss": torch.tensor(3.0),
            "origin_loss": torch.tensor(1.0),
            "fraction_loss": torch.tensor(0.5),
            "slot_loss": torch.tensor(0.25),
            "count_loss": torch.tensor(0.75),
            "fraction_mse": torch.tensor(0.2),
            "fraction_mae": torch.tensor(0.1),
            "pred_class": torch.tensor([0, 2, 3, 1, 1]),
            "true_class": torch.tensor([0, 1, 3, 1, 2]),
            "slot_pred": torch.tensor([True, False, True]),
            "slot_target": torch.tensor([1.0, 0.0, 0.0]),
            "count_target": torch.tensor(2),
            "count_pred": torch.tensor(3),
            "slot_count_pred": torch.tensor(2),
            "num_hits": 5,
        }
        old_totals = empty_slot_metric_totals(num_hit_classes=4, num_count_classes=4)
        new_totals = empty_slot_metric_totals(num_hit_classes=4, num_count_classes=4)

        _old_slot_update(old_totals, losses)
        update_slot_metric_totals(new_totals, losses)

        for key in (
            "loss_sum",
            "origin_loss_sum",
            "fraction_loss_sum",
            "slot_loss_sum",
            "count_loss_sum",
            "fraction_mse_sum",
            "fraction_mae_sum",
            "correct_hits",
            "hits",
            "events",
            "slot_correct",
            "slot_total",
            "slot_exact_correct",
            "count_correct",
            "slot_count_correct",
            "count_total_by_true",
            "count_correct_by_true",
        ):
            self.assertEqual(old_totals[key], new_totals[key])
        self.assertTrue(torch.equal(old_totals["hit_confusion"], new_totals["hit_confusion"]))
        self.assertTrue(torch.equal(old_totals["count_confusion"], new_totals["count_confusion"]))
        self.assertEqual(finalize_slot_metrics(old_totals), finalize_slot_metrics(new_totals))

    def test_hit_confusion_plot_uses_upper_origin(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "confusion.png"
            fig, ax = plot_confusion_matrix(
                confusion=[[3, 0, 0], [0, 2, 0], [0, 0, 1]],
                valid_labels=[1, 2, 3],
                output_path=output_path,
                title="hit confusion",
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(ax.images[0].origin, "upper")
            fig.clear()

    def test_event_count_confusion_plot_uses_upper_origin(self):
        fig, ax = plot_event_count_confusion_matrix(
            y_true=[1, 2, 3],
            y_pred=[1, 2, 3],
            labels=[1, 2, 3],
        )
        self.assertEqual(ax.images[0].origin, "upper")
        fig.clear()


if __name__ == "__main__":
    unittest.main()
