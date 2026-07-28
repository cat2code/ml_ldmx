# ml_ldmx/src/ml_ldmx/data/tensorize.py

import awkward as ak
import numpy as np
import torch

"""
Utilities for converting jagged ECal hit arrays into padded tensors.
Current implementation is simple and readable, not yet optimized.
"""


ECAL_ENERGY_TRANSFORMS = ("raw", "log1p")
TPAD_PE_TRANSFORMS = ("raw", "log1p")
DOMINANT_ORIGIN_TARGET_RULE = "max-summed-edep-by-origin-v1"
LEGACY_DOMINANT_ORIGIN_TARGET_RULE = "max-individual-edep-contribution-v1"
HARD_ORIGIN_TARGET_RULES = (
    DOMINANT_ORIGIN_TARGET_RULE,
    LEGACY_DOMINANT_ORIGIN_TARGET_RULE,
)


def transform_ecal_energy(energy, mode="raw"):
    """Apply the configured reconstructed-energy input transform."""
    if mode not in ECAL_ENERGY_TRANSFORMS:
        raise ValueError(
            f"Unknown ECal energy transform {mode!r}; expected one of {ECAL_ENERGY_TRANSFORMS}."
        )
    if mode == "raw":
        return energy
    if mode == "log1p":
        return torch.log1p(energy.clamp_min(0.0))
    raise AssertionError(f"Unhandled ECal energy transform: {mode}")


def transform_tpad_pe(pe, mode="raw"):
    """Apply the configured TriggerPadTracks pe input transform."""
    if mode not in TPAD_PE_TRANSFORMS:
        raise ValueError(
            f"Unknown TriggerPadTracks pe transform {mode!r}; expected one of {TPAD_PE_TRANSFORMS}."
        )
    if mode == "raw":
        return pe
    if mode == "log1p":
        return torch.log1p(pe.clamp_min(0.0))
    raise AssertionError(f"Unhandled TriggerPadTracks pe transform: {mode}")


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


def tensorize_ecal_event(event, ecal_energy_transform="raw"):
    """
    Return:
      x   : [N, F]
      pos : [N, 3]
    """
    x_vals = _as_tensor(event["x"], torch.float32)
    y_vals = _as_tensor(event["y"], torch.float32)
    z_vals = _as_tensor(event["z"], torch.float32)
    e_vals = transform_ecal_energy(
        _as_tensor(event["energy"], torch.float32),
        mode=ecal_energy_transform,
    )

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


def _summed_energy_by_origin(edeps, origins):
    """Return total deposited energy for each origin represented in one hit."""
    if len(edeps) != len(origins):
        raise ValueError(
            f"Found {len(edeps)} edep contributions but {len(origins)} origin contributions."
        )

    energy_by_origin = {}
    for edep, origin in zip(edeps, origins):
        energy = float(edep)
        if not np.isfinite(energy):
            raise ValueError(f"Deposited-energy contribution must be finite, found {energy}.")
        if energy < 0.0:
            raise ValueError(
                f"Deposited-energy contribution must be non-negative, found {energy}."
            )
        origin = int(origin)
        energy_by_origin[origin] = energy_by_origin.get(origin, 0.0) + energy
    return energy_by_origin


def _dominant_origin_from_summed_energy(edeps, origins, label_order=()):
    """Select the origin with the largest summed energy, with deterministic ties."""
    energy_by_origin = _summed_energy_by_origin(edeps, origins)
    if not energy_by_origin:
        raise ValueError("Cannot select a dominant origin from an empty contribution list.")
    if sum(energy_by_origin.values()) <= 0.0:
        raise ValueError("Cannot select a dominant origin from non-positive total energy.")

    priority = {int(label): idx for idx, label in enumerate(label_order)}
    fallback_priority = len(priority)
    return min(
        energy_by_origin,
        key=lambda origin: (
            -energy_by_origin[origin],
            priority.get(origin, fallback_priority),
            origin,
        ),
    )


def _dominant_origin(
    edeps,
    origins,
    hard_origin_target_rule,
    label_order=(),
):
    if hard_origin_target_rule == DOMINANT_ORIGIN_TARGET_RULE:
        return _dominant_origin_from_summed_energy(
            edeps,
            origins,
            label_order=label_order,
        )
    if hard_origin_target_rule == LEGACY_DOMINANT_ORIGIN_TARGET_RULE:
        return int(origins[int(np.argmax(edeps))])
    raise ValueError(
        f"Unknown hard-origin target rule {hard_origin_target_rule!r}; "
        f"expected one of {HARD_ORIGIN_TARGET_RULES}."
    )


def dominant_origin_class_labels(
    event,
    valid_labels=(1, 2, 3),
    filter_noise=True,
    supervise_noise=False,
    hard_origin_target_rule=DOMINANT_ORIGIN_TARGET_RULE,
):
    """
    Build per-hit class labels using the selected versioned hard-target rule.

    The default rule groups contributions by origin and selects the largest
    total deposited energy. The physical labels are origin IDs in valid_labels.
    Returned class labels are zero-based for PyTorch losses by default. With
    ``supervise_noise``, retained ``noise_flag`` hits receive physical/class
    label 0 for the advanced slot-model background output; non-noise physical
    origin IDs are retained for later canonical slot mapping.
    """
    if supervise_noise and filter_noise:
        raise ValueError("supervise_noise requires filter_noise=False so labelled noise hits are retained.")
    if hard_origin_target_rule not in HARD_ORIGIN_TARGET_RULES:
        raise ValueError(
            f"Unknown hard-origin target rule {hard_origin_target_rule!r}; "
            f"expected one of {HARD_ORIGIN_TARGET_RULES}."
        )

    label_offset = 1 if supervise_noise else 0
    label_to_class = {label: idx + label_offset for idx, label in enumerate(valid_labels)}
    if supervise_noise:
        label_to_class[0] = 0
    keep_indices = []
    physical_labels = []
    class_labels = []
    selected_noise_flags = []
    origin_id_labels = []

    noise_flags = event.get("noise_flag", [False] * len(event["x"]))
    hit_ids = event.get("hit_id", list(range(len(event["x"]))))

    for ihit, (edeps, origins, is_noise) in enumerate(
        zip(event["edep_contribs"], event["origin_id_contribs"], noise_flags)
    ):
        is_noise = bool(is_noise)
        if filter_noise and is_noise:
            continue

        if supervise_noise and is_noise:
            keep_indices.append(ihit)
            physical_labels.append(0)
            class_labels.append(0)
            selected_noise_flags.append(True)
            if len(edeps) > 0 and len(edeps) == len(origins):
                origin_id_labels.append(
                    _dominant_origin(
                        edeps,
                        origins,
                        hard_origin_target_rule=hard_origin_target_rule,
                        label_order=valid_labels,
                    )
                )
            else:
                origin_id_labels.append(-1)
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

        physical_label = _dominant_origin(
            edeps,
            origins,
            hard_origin_target_rule=hard_origin_target_rule,
            label_order=valid_labels,
        )
        if physical_label not in label_to_class:
            raise ValueError(
                f"Hit {hit_ids[ihit]} has dominant origin label {physical_label}, "
                f"but the configured labels are {tuple(valid_labels)}."
            )

        keep_indices.append(ihit)
        physical_labels.append(physical_label)
        class_labels.append(label_to_class[physical_label])
        selected_noise_flags.append(False)
        origin_id_labels.append(physical_label)

    if not keep_indices:
        raise ValueError("No ECal hits remain after applying label/noise selection.")

    labels = {
        "keep_indices": torch.tensor(keep_indices, dtype=torch.long),
        "physical_labels": torch.tensor(physical_labels, dtype=torch.long),
        "class_labels": torch.tensor(class_labels, dtype=torch.long),
        "label_to_class": label_to_class,
        "class_to_label": {idx: label for label, idx in label_to_class.items()},
        "hard_origin_target_rule": hard_origin_target_rule,
    }
    if supervise_noise:
        labels["is_noise_target"] = torch.tensor(selected_noise_flags, dtype=torch.bool)
        labels["origin_id_labels"] = torch.tensor(origin_id_labels, dtype=torch.long)
    return labels


def origin_energy_fraction_targets(
    event,
    keep_indices,
    valid_labels=(1, 2, 3),
    is_noise_target=None,
):
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
    if is_noise_target is not None:
        is_noise_target = torch.as_tensor(is_noise_target, dtype=torch.bool).reshape(-1)
        if is_noise_target.shape[0] != len(keep_indices):
            raise ValueError("is_noise_target must align with kept ECal hits for fraction targets.")

    for row_idx, ihit in enumerate(keep_indices):
        if is_noise_target is not None and bool(is_noise_target[row_idx].item()):
            continue
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

        energy_by_origin = _summed_energy_by_origin(edeps, origins)
        total_edep = float(sum(energy_by_origin.values()))
        if total_edep <= 0.0:
            raise ValueError(
                f"Hit {hit_ids[ihit]} has non-positive total deposited energy "
                f"({total_edep}); cannot normalize origin fractions."
            )

        for origin, origin_edep in energy_by_origin.items():
            column = label_to_column.get(origin)
            if column is not None:
                targets[row_idx, column] = origin_edep / total_edep

    return targets


def tensorize_ecal_node_classification(
    event,
    valid_labels=(1, 2, 3),
    filter_noise=True,
    supervise_noise=False,
    ecal_energy_transform="raw",
    hard_origin_target_rule=DOMINANT_ORIGIN_TARGET_RULE,
):
    x, pos = tensorize_ecal_event(event, ecal_energy_transform=ecal_energy_transform)
    raw_energy = _as_tensor(event["energy"], torch.float32)
    labels = dominant_origin_class_labels(
        event,
        valid_labels=valid_labels,
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        hard_origin_target_rule=hard_origin_target_rule,
    )
    keep_indices = labels["keep_indices"]
    tensors = {
        "x": x[keep_indices],
        "pos": pos[keep_indices],
        "y": labels["class_labels"],
        "physical_y": labels["physical_labels"],
        "ecal_input_energy": x[keep_indices, 3].clone(),
        "ecal_raw_energy": raw_energy[keep_indices].clone(),
        "keep_indices": keep_indices,
        "label_to_class": labels["label_to_class"],
        "class_to_label": labels["class_to_label"],
        "hard_origin_target_rule": labels["hard_origin_target_rule"],
    }
    if supervise_noise:
        tensors["is_noise_target"] = labels["is_noise_target"]
        tensors["origin_id_y"] = labels["origin_id_labels"]
    return tensors


def tensorize_trigger_pad_tracks(event, tpad_pe_transform="raw"):
    """
    Return TriggerPadTracks context features with shape [N_tpad, 2].

    Columns are [centroid, pe]. The centroid_ leaf is treated as the relevant
    1D y-like coordinate for this detector context. ``tpad_pe_transform``
    controls only the pe input feature.
    """

    trigger_pad_tracks = event.get("trigger_pad_tracks", {})
    centroid = trigger_pad_tracks.get("centroid", event.get("tpad_centroid"))
    pe = trigger_pad_tracks.get("pe", event.get("tpad_pe"))

    centroid = _as_1d_float_tensor(centroid)
    pe = transform_tpad_pe(_as_1d_float_tensor(pe), mode=tpad_pe_transform)

    if centroid.numel() == 0 and pe.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if centroid.numel() != pe.numel():
        raise ValueError(
            f"TriggerPadTracks centroid and pe lengths differ: "
            f"{centroid.numel()} vs {pe.numel()}."
        )

    return torch.stack([centroid, pe], dim=1).to(dtype=torch.float32)


def tensorize_ecal_with_triggerpad_context(
    event,
    valid_labels=(1, 2, 3),
    filter_noise=True,
    supervise_noise=False,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
    hard_origin_target_rule=DOMINANT_ORIGIN_TARGET_RULE,
):
    """
    Build one ECal + TriggerPadTracks node tensor for context-aware models.

    Combined features are:
        [is_ecal, is_tpad] + [ecal_x, ecal_y, ecal_z, ecal_energy] + [tpad_centroid, tpad_pe]

    ``ecal_energy_transform`` controls only the reconstructed ECal energy input
    feature. ``tpad_pe_transform`` controls only the TriggerPadTracks pe input
    feature. Truth deposited-energy targets remain in physical units.

    Labels are returned only for selected ECal nodes. TriggerPadTracks nodes are
    context tokens/nodes and should be masked out of the supervised loss.
    """

    ecal = tensorize_ecal_node_classification(
        event,
        valid_labels=valid_labels,
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        ecal_energy_transform=ecal_energy_transform,
        hard_origin_target_rule=hard_origin_target_rule,
    )
    tpad = tensorize_trigger_pad_tracks(event, tpad_pe_transform=tpad_pe_transform)
    trigger_pad_tracks = event.get("trigger_pad_tracks", {})
    tpad_raw_pe = _as_1d_float_tensor(
        trigger_pad_tracks.get("pe", event.get("tpad_pe"))
    )

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

    tensors = {
        "x": x,
        "ecal_pos": ecal["pos"],
        "pos": ecal["pos"],
        "ecal_input_energy": ecal["ecal_input_energy"],
        "ecal_raw_energy": ecal["ecal_raw_energy"],
        "tpad": tpad,
        "tpad_raw_pe": tpad_raw_pe,
        "ecal_mask": ecal_mask,
        "tpad_mask": tpad_mask,
        "y": ecal["y"],
        "physical_y": ecal["physical_y"],
        "keep_indices": ecal["keep_indices"],
        "label_to_class": ecal["label_to_class"],
        "class_to_label": ecal["class_to_label"],
        "hard_origin_target_rule": ecal["hard_origin_target_rule"],
    }
    for key in ("is_noise_target", "origin_id_y"):
        if key in ecal:
            tensors[key] = ecal[key]
    return tensors


def ecal_hits_to_padded_tensor(arrays, vector_branches, max_hits=256, ecal_energy_transform="raw"):
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
        ei = transform_ecal_energy(
            torch.as_tensor(ak.to_numpy(e[i]), dtype=torch.float32),
            mode=ecal_energy_transform,
        ).numpy()

        n_hits = min(len(xi), max_hits)
        if n_hits == 0:
            continue

        features[i, :n_hits, 0] = xi[:n_hits]
        features[i, :n_hits, 1] = yi[:n_hits]
        features[i, :n_hits, 2] = zi[:n_hits]
        features[i, :n_hits, 3] = ei[:n_hits]
        mask[i, :n_hits] = True

    return torch.from_numpy(features), torch.from_numpy(mask)
