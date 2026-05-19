import torch

from mldmx.train.batching import chunks
from mldmx.train.ecal_tpad_slot_model import (
    compute_event_losses,
    empty_slot_metric_totals,
    event_prediction_record,
    finalize_slot_metrics,
    update_slot_metric_totals,
)


@torch.no_grad()
def evaluate(model, events, indices, args, device, split_name, collect_predictions=False):
    model.eval()
    totals = empty_slot_metric_totals(
        num_hit_classes=args.max_electrons + 1,
        num_count_classes=args.max_electrons + 1,
    )
    predictions = []

    for batch in chunks(indices, args.batch_size):
        for event_idx in batch:
            losses = compute_event_losses(model, events[event_idx], device, args)
            update_slot_metric_totals(totals, losses)
            if collect_predictions:
                predictions.append(event_prediction_record(event_idx, events[event_idx], losses))

    metrics = finalize_slot_metrics(totals, prefix=f"{split_name}_")
    metrics[f"{split_name}_hit_confusion"] = totals["hit_confusion"].tolist()
    metrics[f"{split_name}_count_confusion"] = totals["count_confusion"].tolist()
    return metrics, predictions
