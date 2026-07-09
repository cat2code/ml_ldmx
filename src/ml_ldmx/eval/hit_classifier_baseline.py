"""Evaluation utilities for maintained hit-origin classifier baselines."""

import torch
import torch.nn.functional as F

from ml_ldmx.eval.event_diagnostics import event_diagnostic_record
from ml_ldmx.train.batching import chunks
from ml_ldmx.train.hit_classifier_batching import (
    collate_hit_classifier_batch,
    event_views_from_indices,
    hit_classifier_batch_kind,
)
from ml_ldmx.train.hit_classifier_baseline import (
    compute_batch_losses,
    compute_event_losses,
    empty_metric_totals,
    finalize_metrics,
    update_metric_totals,
)


@torch.no_grad()
def evaluate(model, events, indices, view_fn, args, device, split_name):
    model.eval()
    batch_kind = hit_classifier_batch_kind(model)
    totals = empty_metric_totals(num_classes=len(args.valid_labels))
    ordered_indices = (
        events.order_indices_for_access(indices)
        if hasattr(events, "order_indices_for_access")
        else indices
    )
    for batch in chunks(ordered_indices, args.batch_size):
        if batch_kind is None:
            for event_idx in batch:
                losses = compute_event_losses(model, events[event_idx], view_fn, device)
                update_metric_totals(totals, losses)
        else:
            views = event_views_from_indices(events, batch, view_fn)
            losses = compute_batch_losses(model, views, batch_kind, device)
            update_metric_totals(totals, losses)
    return finalize_metrics(totals, prefix=f"{split_name}_")


def _metadata_value(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    return str(value)


def _event_metric_record(
    event_idx,
    split_position,
    view,
    true_class,
    pred_class,
    loss,
    logits=None,
    centroid_radius_mm=25.0,
):
    record = event_diagnostic_record(
        event_idx=event_idx,
        split_position=split_position,
        view=view,
        true_class=true_class,
        pred_class=pred_class,
        loss=loss,
        logits=logits,
        centroid_radius_mm=centroid_radius_mm,
    )
    record.update(
        {
            "source_file": _metadata_value(view.get("source_file")),
            "source_entry": _metadata_value(view.get("source_entry")),
            "source_label": _metadata_value(view.get("source_label")),
            "electron_count": _metadata_value(view.get("electron_count")),
            "target_label_order": _metadata_value(view.get("target_label_order")),
        }
    )
    return record


@torch.no_grad()
def collect_event_metrics(model, events, indices, view_fn, args, device):
    """Collect hit-level accuracy summarized per event for a split."""
    model.eval()
    batch_kind = hit_classifier_batch_kind(model)
    centroid_radius_mm = float(getattr(args, "event_diagnostic_radius_mm", 25.0))
    split_position = {int(event_idx): position for position, event_idx in enumerate(indices)}
    ordered_indices = (
        events.order_indices_for_access(indices)
        if hasattr(events, "order_indices_for_access")
        else indices
    )
    records = []
    for batch_indices in chunks(ordered_indices, args.batch_size):
        if batch_kind is None:
            for event_idx in batch_indices:
                losses = compute_event_losses(model, events[event_idx], view_fn, device)
                records.append(
                    _event_metric_record(
                        event_idx=event_idx,
                        split_position=split_position[int(event_idx)],
                        view=losses["view"],
                        true_class=losses["true_class"],
                        pred_class=losses["pred_class"],
                        loss=losses["total_loss"],
                        logits=losses.get("supervised_logits"),
                        centroid_radius_mm=centroid_radius_mm,
                    )
                )
            continue

        views = event_views_from_indices(events, batch_indices, view_fn)
        hit_batch = collate_hit_classifier_batch(views, batch_kind).to(device)
        if hit_batch.kind == "padded":
            logits = model(hit_batch.x, key_padding_mask=~hit_batch.valid_mask)
            for row, (event_idx, view) in enumerate(zip(batch_indices, views)):
                mask = hit_batch.supervised_mask[row]
                target = hit_batch.target[row][mask]
                event_logits = logits[row][mask]
                pred_class = event_logits.argmax(dim=1)
                loss = F.cross_entropy(event_logits, target) if target.numel() else None
                records.append(
                    _event_metric_record(
                        event_idx=event_idx,
                        split_position=split_position[int(event_idx)],
                        view=view,
                        true_class=target,
                        pred_class=pred_class,
                        loss=loss,
                        logits=event_logits,
                        centroid_radius_mm=centroid_radius_mm,
                    )
                )
        elif hit_batch.kind == "graph":
            logits = model(hit_batch.x, batch=hit_batch.batch_index)
            for row, (event_idx, view) in enumerate(zip(batch_indices, views)):
                mask = (hit_batch.batch_index == row) & hit_batch.supervised_mask
                target = hit_batch.target[mask]
                event_logits = logits[mask]
                pred_class = event_logits.argmax(dim=1)
                loss = F.cross_entropy(event_logits, target) if target.numel() else None
                records.append(
                    _event_metric_record(
                        event_idx=event_idx,
                        split_position=split_position[int(event_idx)],
                        view=view,
                        true_class=target,
                        pred_class=pred_class,
                        loss=loss,
                        logits=event_logits,
                        centroid_radius_mm=centroid_radius_mm,
                    )
                )
        else:
            raise ValueError(f"Unsupported hit-classifier batch kind: {hit_batch.kind!r}.")
    return sorted(records, key=lambda record: record["split_position"])
