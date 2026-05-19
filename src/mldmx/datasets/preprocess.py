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
