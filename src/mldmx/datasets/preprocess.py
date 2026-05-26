import torch


def normalize_continuous_features(tensor_events, train_indices, first_continuous_col=2):
    train_x = torch.cat(
        [tensor_events[idx]["x"][:, first_continuous_col:] for idx in train_indices],
        dim=0,
    )
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)

    for event in tensor_events:
        x = event["x"].clone()
        x[:, first_continuous_col:] = (x[:, first_continuous_col:] - mean) / std
        event["x"] = x

    return {
        "first_continuous_col": first_continuous_col,
        "mean": mean,
        "std": std,
    }


def fit_continuous_feature_normalization(tensor_events, train_indices, first_continuous_col=2):
    """Fit normalization statistics without materializing all training nodes at once."""
    ordered_indices = (
        tensor_events.order_indices_for_access(train_indices)
        if hasattr(tensor_events, "order_indices_for_access")
        else train_indices
    )
    total = None
    total_squared = None
    count = 0
    for idx in ordered_indices:
        values = tensor_events[idx]["x"][:, first_continuous_col:].to(dtype=torch.float64)
        if values.numel() == 0:
            continue
        value_sum = values.sum(dim=0)
        value_squared_sum = (values * values).sum(dim=0)
        total = value_sum if total is None else total + value_sum
        total_squared = value_squared_sum if total_squared is None else total_squared + value_squared_sum
        count += int(values.shape[0])
    if count < 2 or total is None:
        raise ValueError("Need at least two nodes in the training split to fit feature normalization.")
    mean = total / count
    variance = ((total_squared - count * mean * mean) / (count - 1)).clamp_min(1e-12)
    return {
        "first_continuous_col": first_continuous_col,
        "mean": mean.to(dtype=torch.float32),
        "std": variance.sqrt().to(dtype=torch.float32).clamp_min(1e-6),
    }


def normalize_event_continuous_features(event, feature_norm):
    """Apply pre-fitted normalization to one event while leaving cached tensors unchanged."""
    x = event["x"].clone()
    first_col = int(feature_norm["first_continuous_col"])
    x[:, first_col:] = (x[:, first_col:] - feature_norm["mean"]) / feature_norm["std"]
    event["x"] = x
    return event
