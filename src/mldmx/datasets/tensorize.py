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


def _as_1d_float_tensor(values):
    if values is None:
        return torch.empty((0,), dtype=torch.float32)
    if isinstance(values, torch.Tensor):
        return values.to(dtype=torch.float32).reshape(-1)
    if isinstance(values, ak.Array):
        values = ak.to_list(values)

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return torch.empty((0,), dtype=torch.float32)
    return torch.as_tensor(array.reshape(-1), dtype=torch.float32)


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


def origin_energy_fraction_targets(event, keep_indices, valid_labels=(1, 2, 3)):
    """
    Build soft per-hit origin-composition targets from deposited energy fractions.

    The returned tensor has one row per kept ECal hit and one column per origin in
    valid_labels. Contributions from other origins are included in the deposited
    energy denominator but ignored in the numerator.
    """
    label_to_column = {label: idx for idx, label in enumerate(valid_labels)}
    for key in ("edep_contribs", "origin_id_contribs"):
        if key not in event:
            raise ValueError(
                f"Event is missing '{key}'; cannot build origin energy-fraction targets."
            )

    hit_ids = event.get("hit_id", list(range(len(event["edep_contribs"]))))

    if isinstance(keep_indices, torch.Tensor):
        keep_indices = keep_indices.detach().cpu().tolist()
    elif isinstance(keep_indices, ak.Array):
        keep_indices = ak.to_list(keep_indices)

    targets = torch.zeros((len(keep_indices), len(valid_labels)), dtype=torch.float32)

    for row_idx, ihit in enumerate(keep_indices):
        ihit = int(ihit)
        if ihit < 0 or ihit >= len(event["edep_contribs"]):
            raise ValueError(
                f"keep_indices contains hit index {ihit}, but event has "
                f"{len(event['edep_contribs'])} contribution rows."
            )
        edeps = event["edep_contribs"][ihit]
        origins = event["origin_id_contribs"][ihit]

        if len(edeps) == 0:
            raise ValueError(
                f"Hit {hit_ids[ihit]} has no energy contributions; cannot build "
                "origin energy-fraction targets."
            )
        if len(edeps) != len(origins):
            raise ValueError(
                f"Hit {hit_ids[ihit]} has {len(edeps)} edep contributions but "
                f"{len(origins)} origin contributions."
            )

        total_edep = float(sum(float(edep) for edep in edeps))
        if total_edep <= 0.0:
            raise ValueError(
                f"Hit {hit_ids[ihit]} has non-positive total deposited energy "
                f"({total_edep}); cannot normalize origin fractions."
            )

        for edep, origin in zip(edeps, origins):
            column = label_to_column.get(int(origin))
            if column is not None:
                targets[row_idx, column] += float(edep) / total_edep

    return targets


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


def tensorize_trigger_pad_tracks(event):
    """
    Return TriggerPadTracks context features with shape [N_tpad, 2].

    Columns are [centroid, pe]. The centroid_ leaf is treated as the relevant
    1D y-like coordinate for this detector context.
    """

    trigger_pad_tracks = event.get("trigger_pad_tracks", {})
    centroid = trigger_pad_tracks.get("centroid", event.get("tpad_centroid"))
    pe = trigger_pad_tracks.get("pe", event.get("tpad_pe"))

    centroid = _as_1d_float_tensor(centroid)
    pe = _as_1d_float_tensor(pe)

    if centroid.numel() == 0 and pe.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if centroid.numel() != pe.numel():
        raise ValueError(
            f"TriggerPadTracks centroid and pe lengths differ: "
            f"{centroid.numel()} vs {pe.numel()}."
        )

    return torch.stack([centroid, pe], dim=1).to(dtype=torch.float32)


def tensorize_ecal_with_triggerpad_context(event, valid_labels=(1, 2, 3), filter_noise=True):
    """
    Build one ECal + TriggerPadTracks node tensor for context-aware models.

    Combined features are:
        [is_ecal, is_tpad] + [ecal_x, ecal_y, ecal_z, ecal_energy] + [tpad_centroid, tpad_pe]

    Labels are returned only for selected ECal nodes. TriggerPadTracks nodes are
    context tokens/nodes and should be masked out of the supervised loss.
    """

    ecal = tensorize_ecal_node_classification(
        event,
        valid_labels=valid_labels,
        filter_noise=filter_noise,
    )
    tpad = tensorize_trigger_pad_tracks(event)

    ecal_x = ecal["x"]
    num_ecal = ecal_x.shape[0]
    num_tpad = tpad.shape[0]
    ecal_feature_dim = ecal_x.shape[1]

    ecal_nodes = torch.cat(
        [
            torch.ones((num_ecal, 1), dtype=torch.float32),
            torch.zeros((num_ecal, 1), dtype=torch.float32),
            ecal_x.to(dtype=torch.float32),
            torch.zeros((num_ecal, 2), dtype=torch.float32),
        ],
        dim=1,
    )

    if num_tpad == 0:
        tpad_nodes = torch.empty((0, ecal_nodes.shape[1]), dtype=torch.float32)
    else:
        tpad_nodes = torch.cat(
            [
                torch.zeros((num_tpad, 1), dtype=torch.float32),
                torch.ones((num_tpad, 1), dtype=torch.float32),
                torch.zeros((num_tpad, ecal_feature_dim), dtype=torch.float32),
                tpad.to(dtype=torch.float32),
            ],
            dim=1,
        )

    x = torch.cat([ecal_nodes, tpad_nodes], dim=0)
    ecal_mask = torch.zeros((x.shape[0],), dtype=torch.bool)
    ecal_mask[:num_ecal] = True
    tpad_mask = ~ecal_mask

    return {
        "x": x,
        "ecal_pos": ecal["pos"],
        "pos": ecal["pos"],
        "tpad": tpad,
        "ecal_mask": ecal_mask,
        "tpad_mask": tpad_mask,
        "y": ecal["y"],
        "physical_y": ecal["physical_y"],
        "keep_indices": ecal["keep_indices"],
        "label_to_class": ecal["label_to_class"],
        "class_to_label": ecal["class_to_label"],
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
