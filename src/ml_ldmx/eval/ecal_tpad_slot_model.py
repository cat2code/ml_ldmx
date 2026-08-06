import torch

from ml_ldmx.train.batching import chunks
from ml_ldmx.train.ecal_tpad_slot_model import (
    batch_prediction_records,
    compute_batch_losses,
    empty_slot_metric_totals,
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

    ordered_indices = (
        events.order_indices_for_access(indices)
        if hasattr(events, "order_indices_for_access")
        else indices
    )
    for batch in chunks(ordered_indices, args.batch_size):
        batch_events = [events[event_idx] for event_idx in batch]
        losses = compute_batch_losses(model, batch_events, device, args)
        update_slot_metric_totals(totals, losses)
        if collect_predictions:
            predictions.extend(batch_prediction_records(batch, batch_events, losses))

    metrics = finalize_slot_metrics(totals, prefix=f"{split_name}_")
    metrics[f"{split_name}_hit_confusion"] = totals["hit_confusion"].tolist()
    metrics[f"{split_name}_count_confusion"] = totals["count_confusion"].tolist()
    return metrics, predictions
