# mldmx/src/mldmx/data/tensorize.py

import awkward as ak
import numpy as np
import torch

"""
Utilities for converting jagged ECal hit arrays into padded tensors.
Current implementation is simple and readable, not yet optimized.
"""


def _as_tensor(values, dtype):
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype)
    if isinstance(values, ak.Array):
        values = ak.to_numpy(values)
    return torch.as_tensor(values, dtype=dtype)


def tensorize_ecal_event(event):
    """
    Return:
      x   : [N, F]
      pos : [N, 3]
    """
    x_vals = _as_tensor(event["x"], torch.float32)
    y_vals = _as_tensor(event["y"], torch.float32)
    z_vals = _as_tensor(event["z"], torch.float32)
    e_vals = _as_tensor(event["energy"], torch.float32)

    pos = torch.stack([x_vals, y_vals, z_vals], dim=1)
    x = torch.stack([x_vals, y_vals, z_vals, e_vals], dim=1)

    return x, pos


def tensorize_ecal_truth(event):
    """
    Convert per-hit truth fields that have fixed length into tensors.

    Variable-length contribution fields remain Python lists because each hit can
    have a different number of contributing particles.
    """
    truth = {}
    if "hit_id" in event:
        truth["hit_id"] = _as_tensor(event["hit_id"], torch.long)
    if "noise_flag" in event:
        truth["noise_flag"] = _as_tensor(event["noise_flag"], torch.bool)
    if "n_contribs" in event:
        truth["n_contribs"] = _as_tensor(event["n_contribs"], torch.long)

    for key in ["track_id_contribs", "edep_contribs", "origin_id_contribs"]:
        if key in event:
            truth[key] = event[key]

    return truth


def ecal_hits_to_padded_tensor(arrays, vector_branches, max_hits=256):
    x = arrays[vector_branches["x"]]
    y = arrays[vector_branches["y"]]
    z = arrays[vector_branches["z"]]
    e = arrays[vector_branches["energy"]]

    n_events = len(x)
    features = np.zeros((n_events, max_hits, 4), dtype=np.float32)
    mask = np.zeros((n_events, max_hits), dtype=bool)

    for i in range(n_events):
        xi = ak.to_numpy(x[i])
        yi = ak.to_numpy(y[i])
        zi = ak.to_numpy(z[i])
        ei = ak.to_numpy(e[i])

        n_hits = min(len(xi), max_hits)
        if n_hits == 0:
            continue

        features[i, :n_hits, 0] = xi[:n_hits]
        features[i, :n_hits, 1] = yi[:n_hits]
        features[i, :n_hits, 2] = zi[:n_hits]
        features[i, :n_hits, 3] = ei[:n_hits]
        mask[i, :n_hits] = True

    return torch.from_numpy(features), torch.from_numpy(mask)
