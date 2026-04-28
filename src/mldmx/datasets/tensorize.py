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


def dominant_origin_class_labels(event, valid_labels=(1, 2, 3), filter_noise=True):
    """
    Build per-hit class labels from the dominant deposited-energy contribution.

    The physical labels are origin IDs in valid_labels. Returned class labels are
    zero-based for PyTorch losses, with label 1 -> class 0, label 2 -> class 1,
    and label 3 -> class 2 by default.
    """
    label_to_class = {label: idx for idx, label in enumerate(valid_labels)}
    keep_indices = []
    physical_labels = []
    class_labels = []

    noise_flags = event.get("noise_flag", [False] * len(event["x"]))
    hit_ids = event.get("hit_id", list(range(len(event["x"]))))

    for ihit, (edeps, origins, is_noise) in enumerate(
        zip(event["edep_contribs"], event["origin_id_contribs"], noise_flags)
    ):
        if filter_noise and bool(is_noise):
            continue

        if len(edeps) == 0:
            raise ValueError(
                f"Hit {hit_ids[ihit]} has no energy contributions; cannot assign an origin label."
            )
        if len(edeps) != len(origins):
            raise ValueError(
                f"Hit {hit_ids[ihit]} has {len(edeps)} edep contributions but "
                f"{len(origins)} origin contributions."
            )

        dom = int(np.argmax(edeps))
        physical_label = int(origins[dom])
        if physical_label not in label_to_class:
            raise ValueError(
                f"Hit {hit_ids[ihit]} has dominant origin label {physical_label}, "
                f"but this prototype only accepts {tuple(valid_labels)}."
            )

        keep_indices.append(ihit)
        physical_labels.append(physical_label)
        class_labels.append(label_to_class[physical_label])

    if not keep_indices:
        raise ValueError("No ECal hits remain after applying label/noise selection.")

    return {
        "keep_indices": torch.tensor(keep_indices, dtype=torch.long),
        "physical_labels": torch.tensor(physical_labels, dtype=torch.long),
        "class_labels": torch.tensor(class_labels, dtype=torch.long),
        "label_to_class": label_to_class,
        "class_to_label": {idx: label for label, idx in label_to_class.items()},
    }


def tensorize_ecal_node_classification(event, valid_labels=(1, 2, 3), filter_noise=True):
    x, pos = tensorize_ecal_event(event)
    labels = dominant_origin_class_labels(
        event,
        valid_labels=valid_labels,
        filter_noise=filter_noise,
    )
    keep_indices = labels["keep_indices"]
    return {
        "x": x[keep_indices],
        "pos": pos[keep_indices],
        "y": labels["class_labels"],
        "physical_y": labels["physical_labels"],
        "keep_indices": keep_indices,
        "label_to_class": labels["label_to_class"],
        "class_to_label": labels["class_to_label"],
    }


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
