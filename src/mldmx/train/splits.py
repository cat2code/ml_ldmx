import torch


def deterministic_split(num_events, seed):
    if num_events < 20:
        raise ValueError(
            f"Need at least 20 events for an 80/15/5 split with a non-empty test set; got {num_events}."
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_events, generator=generator).tolist()
    n_train = int(0.80 * num_events)
    n_val = int(0.15 * num_events)
    n_test = num_events - n_train - n_val
    if n_val == 0 or n_test == 0:
        raise ValueError(f"Split produced empty validation/test sets for {num_events} events.")
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }
