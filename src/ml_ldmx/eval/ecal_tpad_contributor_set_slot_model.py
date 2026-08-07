"""Evaluation loop for the contributor-set slot model."""

import torch

from ml_ldmx.train.batching import chunks
from ml_ldmx.train.ecal_tpad_contributor_set_slot_model import (
    batch_event_prediction_records,
    compute_batch_losses,
    empty_metric_totals,
    finalize_metrics,
    update_metric_totals,
)


@torch.no_grad()
def evaluate(
    model,
    events,
    indices,
    args,
    device,
    split_name,
    *,
    collect_predictions=False,
    max_plot_hits=0,
):
    model.eval()
    totals = empty_metric_totals(model.num_fraction_classes, model.num_support_classes)
    predictions = []
    plot_chunks = {
        "fraction_target": [],
        "fraction_pred": [],
        "support_target": [],
        "support_pred": [],
        "mixed_target": [],
        "mixed_probability": [],
    }
    collected_hits = 0
    ordered = (
        events.order_indices_for_access(indices)
        if hasattr(events, "order_indices_for_access")
        else indices
    )
    for batch_indices in chunks(ordered, args.batch_size):
        batch_events = [events[index] for index in batch_indices]
        losses = compute_batch_losses(model, batch_events, device, args)
        update_metric_totals(totals, losses)
        if collect_predictions:
            predictions.extend(
                batch_event_prediction_records(batch_indices, batch_events, losses)
            )
        if max_plot_hits > collected_hits:
            take = min(max_plot_hits - collected_hits, int(losses["num_hits"]))
            for key in plot_chunks:
                plot_chunks[key].append(losses[key][:take].detach().cpu())
            collected_hits += take

    metrics = finalize_metrics(totals, prefix=f"{split_name}_")
    metrics[f"{split_name}_origin_confusion"] = totals["origin_confusion"].tolist()
    metrics[f"{split_name}_support_confusion"] = totals["support_confusion"].tolist()
    metrics[f"{split_name}_mixed_confusion"] = totals["mixed_confusion"].tolist()
    metrics[f"{split_name}_count_confusion"] = totals["count_confusion"].tolist()
    plot_data = {
        key: torch.cat(values, dim=0) if values else torch.empty(0)
        for key, values in plot_chunks.items()
    }
    return metrics, predictions, plot_data
