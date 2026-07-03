import torch

from ml_ldmx.train.batching import chunks
from ml_ldmx.train.ecal_tpad_mlpf_lite import compute_event_losses
from ml_ldmx.train.metrics import empty_metric_totals, finalize_metrics, update_metric_totals


@torch.no_grad()
def evaluate(model, events, indices, args, device, split_name, collect_plot_samples=False):
    model.eval()
    totals = empty_metric_totals(num_classes=len(args.valid_labels))
    plot_samples = {
        "fraction_target": [],
        "fraction_pred": [],
        "per_hit_fraction_mae": [],
    }
    remaining_plot_hits = args.num_plot_hits if collect_plot_samples else 0

    for batch in chunks(indices, args.batch_size):
        for event_idx in batch:
            losses = compute_event_losses(model, events[event_idx], device, args.lambda_fraction)
            update_metric_totals(totals, losses)

            if remaining_plot_hits > 0:
                take = min(remaining_plot_hits, int(losses["num_hits"]))
                plot_samples["fraction_target"].append(losses["fraction_target"][:take].detach().cpu())
                plot_samples["fraction_pred"].append(losses["fraction_pred"][:take].detach().cpu())
                plot_samples["per_hit_fraction_mae"].append(
                    losses["per_hit_fraction_mae"][:take].detach().cpu()
                )
                remaining_plot_hits -= take

    metrics = finalize_metrics(totals, prefix=f"{split_name}_")
    metrics[f"{split_name}_confusion"] = totals["confusion"].tolist()

    if collect_plot_samples:
        for key, tensors in plot_samples.items():
            plot_samples[key] = torch.cat(tensors, dim=0) if tensors else torch.empty((0,))
        return metrics, plot_samples
    return metrics, None
