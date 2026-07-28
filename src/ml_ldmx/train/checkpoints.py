import torch

from ml_ldmx.datasets.tensorize import (
    DOMINANT_ORIGIN_TARGET_RULE,
    HARD_ORIGIN_TARGET_RULES,
    LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
)


def checkpoint_hard_origin_target_rule(checkpoint):
    """Resolve target semantics while preserving checkpoints written before policy metadata."""
    checkpoint_args = checkpoint.get("args") or {}
    top_level_rule = checkpoint.get("hard_origin_target_rule")
    args_rule = checkpoint_args.get("hard_origin_target_rule")
    if top_level_rule is not None and args_rule is not None and top_level_rule != args_rule:
        raise ValueError(
            "Checkpoint has inconsistent hard-origin target policy metadata: "
            f"{top_level_rule!r} != {args_rule!r}."
        )
    resolved_rule = top_level_rule or args_rule or LEGACY_DOMINANT_ORIGIN_TARGET_RULE
    if resolved_rule not in HARD_ORIGIN_TARGET_RULES:
        raise ValueError(
            f"Checkpoint has unknown hard-origin target rule {resolved_rule!r}; "
            f"expected one of {HARD_ORIGIN_TARGET_RULES}."
        )
    return resolved_rule


def require_matching_hard_origin_target_rule(checkpoint, requested_rule):
    """Reject resume requests that would change the checkpoint's hard-label semantics."""
    checkpoint_rule = checkpoint_hard_origin_target_rule(checkpoint)
    if checkpoint_rule != requested_rule:
        raise ValueError(
            f"Checkpoint hard-origin target rule {checkpoint_rule!r} does not match "
            f"current rule {requested_rule!r}."
        )
    return checkpoint_rule


def checkpoint_state(model, optimizer, scheduler, epoch, args, history, best_val_loss, model_kwargs, feature_norm, splits):
    hard_origin_target_rule = getattr(
        args,
        "hard_origin_target_rule",
        DOMINANT_ORIGIN_TARGET_RULE,
    )
    if hard_origin_target_rule not in HARD_ORIGIN_TARGET_RULES:
        raise ValueError(
            f"Unknown hard-origin target rule {hard_origin_target_rule!r}; "
            f"expected one of {HARD_ORIGIN_TARGET_RULES}."
        )
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
        "hard_origin_target_rule": hard_origin_target_rule,
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
