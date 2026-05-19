import torch


def checkpoint_state(model, optimizer, scheduler, epoch, args, history, best_val_loss, model_kwargs, feature_norm, splits):
    return {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "history": history,
        "args": vars(args),
        "best_val_loss": best_val_loss,
        "model_kwargs": model_kwargs,
        "feature_norm": {
            "first_continuous_col": feature_norm["first_continuous_col"],
            "mean": feature_norm["mean"].detach().cpu().tolist(),
            "std": feature_norm["std"].detach().cpu().tolist(),
        }
        if feature_norm is not None
        else None,
        "splits": splits,
        "valid_labels": tuple(args.valid_labels),
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, args, history, best_val_loss, model_kwargs, feature_norm, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_state(
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            history,
            best_val_loss,
            model_kwargs,
            feature_norm,
            splits,
        ),
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
