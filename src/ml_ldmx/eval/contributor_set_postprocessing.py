"""Coherent postprocessing for contributor-set slot-model outputs."""

import torch


def support_bit_table(max_electrons: int, *, device=None) -> torch.Tensor:
    """Return ``[2**K, K]`` bool rows encoding contributor subsets."""
    if max_electrons <= 0:
        raise ValueError("max_electrons must be positive.")
    values = torch.arange(1 << max_electrons, device=device, dtype=torch.long)
    shifts = torch.arange(max_electrons, device=device, dtype=torch.long)
    return values.unsqueeze(1).bitwise_and(1 << shifts).ne(0)


def fraction_targets_to_support(
    fraction_target: torch.Tensor,
    *,
    contribution_epsilon: float = 0.0,
) -> torch.Tensor:
    """Encode the exact nonzero electron contributors of each hit as a bit mask."""
    if fraction_target.ndim < 2 or fraction_target.shape[-1] < 2:
        raise ValueError("fraction_target must end in [noise, electron_1, ...].")
    if contribution_epsilon < 0:
        raise ValueError("contribution_epsilon must be non-negative.")
    electron_present = fraction_target[..., 1:] > contribution_epsilon
    powers = 1 << torch.arange(
        electron_present.shape[-1],
        device=fraction_target.device,
        dtype=torch.long,
    )
    return (electron_present.to(dtype=torch.long) * powers).sum(dim=-1)


def _as_batched(outputs: dict[str, torch.Tensor]):
    support_logits = outputs["support_logits"]
    fraction_logits = outputs["fraction_logits"]
    slot_logits = outputs["slot_valid_logits"]
    single_event = support_logits.ndim == 2
    if single_event:
        support_logits = support_logits.unsqueeze(0)
        fraction_logits = fraction_logits.unsqueeze(0)
        slot_logits = slot_logits.unsqueeze(0)
    if support_logits.ndim != 3 or fraction_logits.ndim != 3 or slot_logits.ndim != 2:
        raise ValueError("Expected support/fraction token logits and event-level slot logits.")
    if support_logits.shape[:2] != fraction_logits.shape[:2]:
        raise ValueError("Support and fraction logits must align on batch and token dimensions.")
    if support_logits.shape[0] != slot_logits.shape[0]:
        raise ValueError("Slot logits must align with the token-logit batch dimension.")
    return support_logits, fraction_logits, slot_logits, single_event


def postprocess_contributor_set_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    min_electrons: int = 2,
    key_padding_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Derive a coherent reconstructed event from parallel raw model heads.

    The most probable legal prefix of active electron slots determines the
    event count. Contributor sets containing inactive slots are removed. The
    predicted maximum-probability contributor set then gates and renormalizes
    each hit's fractions, producing exact zeros for absent contributors.
    """
    support_logits, fraction_logits, slot_logits, single_event = _as_batched(outputs)
    batch_size, num_tokens, num_support_classes = support_logits.shape
    max_electrons = int(slot_logits.shape[-1])
    if fraction_logits.shape[-1] != max_electrons + 1:
        raise ValueError("Fraction logits must contain noise plus one class per electron slot.")
    if num_support_classes != 1 << max_electrons:
        raise ValueError("Support logits must contain every electron-subset class.")
    if min_electrons < 0 or min_electrons > max_electrons:
        raise ValueError("min_electrons must be in 0..max_electrons.")

    if key_padding_mask is None:
        key_padding_mask = torch.zeros(
            (batch_size, num_tokens),
            dtype=torch.bool,
            device=support_logits.device,
        )
    elif single_event and key_padding_mask.ndim == 1:
        key_padding_mask = key_padding_mask.unsqueeze(0)
    if key_padding_mask.shape != (batch_size, num_tokens):
        raise ValueError(
            f"Expected key_padding_mask [{batch_size}, {num_tokens}], "
            f"got {tuple(key_padding_mask.shape)}."
        )
    key_padding_mask = key_padding_mask.to(device=support_logits.device, dtype=torch.bool)

    slot_probability = torch.sigmoid(slot_logits)
    eps = torch.finfo(slot_probability.dtype).eps
    log_present = slot_probability.clamp(eps, 1.0 - eps).log()
    log_absent = (1.0 - slot_probability).clamp(eps, 1.0).log()
    candidate_counts = torch.arange(
        min_electrons,
        max_electrons + 1,
        device=slot_logits.device,
        dtype=torch.long,
    )
    slot_indices = torch.arange(max_electrons, device=slot_logits.device)
    candidate_masks = slot_indices.unsqueeze(0) < candidate_counts.unsqueeze(1)
    count_scores = (
        candidate_masks.unsqueeze(0) * log_present.unsqueeze(1)
        + (~candidate_masks).unsqueeze(0) * log_absent.unsqueeze(1)
    ).sum(dim=-1)
    count_probability = torch.softmax(count_scores, dim=-1)
    selected_count_index = count_scores.argmax(dim=-1)
    predicted_count = candidate_counts[selected_count_index]
    slot_valid_mask = slot_indices.unsqueeze(0) < predicted_count.unsqueeze(1)

    bits = support_bit_table(max_electrons, device=support_logits.device)
    impossible_support = (
        bits.unsqueeze(0) & ~slot_valid_mask.unsqueeze(1)
    ).any(dim=-1)
    negative_large = torch.finfo(support_logits.dtype).min
    gated_support_logits = support_logits.masked_fill(
        impossible_support.unsqueeze(1),
        negative_large,
    )
    support_probability = torch.softmax(gated_support_logits, dim=-1)
    support_prediction = support_probability.argmax(dim=-1)

    predicted_bits = bits[support_prediction]
    noise_allowed = support_prediction.eq(0).unsqueeze(-1)
    fraction_allowed = torch.cat([noise_allowed, predicted_bits], dim=-1)
    gated_fraction_logits = fraction_logits.masked_fill(~fraction_allowed, negative_large)
    fraction_prediction = torch.softmax(gated_fraction_logits, dim=-1)

    support_cardinality_table = bits.sum(dim=-1).to(dtype=torch.long)
    support_cardinality = support_cardinality_table[support_prediction]
    mixed_classes = support_cardinality_table >= 2
    pure_classes = support_cardinality_table == 1
    mixed_probability = support_probability[..., mixed_classes].sum(dim=-1)
    pure_probability = support_probability[..., pure_classes].sum(dim=-1)
    noise_probability = support_probability[..., 0]
    mixed_prediction = support_cardinality >= 2
    dominant_origin = fraction_prediction.argmax(dim=-1)

    if key_padding_mask.any():
        support_prediction = support_prediction.masked_fill(key_padding_mask, 0)
        support_cardinality = support_cardinality.masked_fill(key_padding_mask, 0)
        mixed_prediction = mixed_prediction.masked_fill(key_padding_mask, False)
        dominant_origin = dominant_origin.masked_fill(key_padding_mask, 0)
        fraction_prediction = fraction_prediction.clone()
        fraction_prediction[key_padding_mask] = 0.0
        fraction_prediction[..., 0][key_padding_mask] = 1.0
        mixed_probability = mixed_probability.masked_fill(key_padding_mask, 0.0)
        pure_probability = pure_probability.masked_fill(key_padding_mask, 0.0)
        noise_probability = noise_probability.masked_fill(key_padding_mask, 1.0)

    processed = {
        "slot_probability": slot_probability,
        "slot_valid_mask": slot_valid_mask,
        "count_probability": count_probability,
        "count_values": candidate_counts,
        "predicted_count": predicted_count,
        "gated_support_logits": gated_support_logits,
        "support_probability": support_probability,
        "support_prediction": support_prediction,
        "support_cardinality": support_cardinality,
        "mixed_probability": mixed_probability,
        "pure_probability": pure_probability,
        "noise_probability": noise_probability,
        "mixed_prediction": mixed_prediction,
        "gated_fraction_logits": gated_fraction_logits,
        "fraction_prediction": fraction_prediction,
        "dominant_origin": dominant_origin,
    }
    if single_event:
        return {key: value.squeeze(0) for key, value in processed.items()}
    return processed
