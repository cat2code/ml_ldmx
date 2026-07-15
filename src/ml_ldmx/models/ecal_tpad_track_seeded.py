"""Track-seeded canonical-slot classifier for ECal hit-origin assignment."""

import math

import torch
import torch.nn as nn


class ECalTpadTrackSeededTransformer(nn.Module):
    """
    Classify ECal hits with event-dependent canonical electron slots.

    The token encoder retains the maintained full-event Transformer backbone.
    Fixed ordered slots then attend first to available TPad tokens and next to
    ECal tokens. A learned null-track token keeps every slot usable when one or
    more expected TPad tracks are absent.
    """

    hit_classifier_batch_kind = "padded"

    def __init__(
        self,
        in_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        out_dim: int = 3,
    ):
        super().__init__()
        if in_dim < 2 or d_model <= 0 or out_dim <= 0:
            raise ValueError("in_dim must include detector flags; d_model and out_dim must be positive.")
        if nhead <= 0 or num_layers <= 0 or dim_feedforward <= 0:
            raise ValueError("nhead, num_layers, and dim_feedforward must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.d_model = int(d_model)

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.type_embedding = nn.Embedding(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.canonical_slots = nn.Parameter(torch.empty(out_dim, d_model))
        self.null_track = nn.Parameter(torch.empty(1, d_model))
        nn.init.normal_(self.canonical_slots, std=0.02)
        nn.init.normal_(self.null_track, std=0.02)

        self.track_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.shower_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.track_norm = nn.LayerNorm(d_model)
        self.shower_norm = nn.LayerNorm(d_model)
        self.slot_ffn_norm = nn.LayerNorm(d_model)
        self.slot_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.slot_dropout = nn.Dropout(dropout)

        self.hit_score_proj = nn.Linear(d_model, d_model, bias=False)
        self.slot_score_proj = nn.Linear(d_model, d_model, bias=False)
        self.base_head = nn.Linear(d_model, out_dim)
        self.class_bias = nn.Parameter(torch.zeros(out_dim))

    def _validate_and_batch(self, x, key_padding_mask):
        if x.ndim not in (2, 3) or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected x with shape [N, {self.in_dim}] or [B, N, {self.in_dim}], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-2] == 0:
            raise ValueError("Expected at least one event token.")

        single_event = x.ndim == 2
        if single_event:
            x = x.unsqueeze(0)
            if key_padding_mask is not None and key_padding_mask.ndim == 1:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        elif key_padding_mask.shape != x.shape[:2]:
            raise ValueError(
                f"Expected key_padding_mask with shape {tuple(x.shape[:2])}, "
                f"got {tuple(key_padding_mask.shape)}."
            )
        else:
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)
        return x, key_padding_mask, single_event

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return canonical-origin logits for every input token."""
        x, key_padding_mask, single_event = self._validate_and_batch(x, key_padding_mask)
        valid_mask = ~key_padding_mask
        ecal_mask = valid_mask & (x[..., 0] > x[..., 1])
        tpad_mask = valid_mask & (x[..., 1] > x[..., 0])
        if not bool(ecal_mask.any(dim=1).all().item()):
            raise ValueError("Every event must contain at least one ECal token.")

        node_type = tpad_mask.to(dtype=torch.long)
        encoded = self.input_proj(x) + self.type_embedding(node_type)
        encoded = self.encoder(encoded, src_key_padding_mask=key_padding_mask)

        batch_size = x.shape[0]
        # MPS multi-head attention requires materialized batched query/key buffers.
        slots = self.canonical_slots.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        null_track = self.null_track.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        track_memory = torch.cat([encoded, null_track], dim=1)
        null_is_valid = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        track_padding_mask = torch.cat([~tpad_mask, null_is_valid], dim=1)
        track_context, _weights = self.track_attention(
            query=slots,
            key=track_memory,
            value=track_memory,
            key_padding_mask=track_padding_mask,
            need_weights=False,
        )
        slots = self.track_norm(slots + self.slot_dropout(track_context))

        shower_context, _weights = self.shower_attention(
            query=slots,
            key=encoded,
            value=encoded,
            key_padding_mask=~ecal_mask,
            need_weights=False,
        )
        slots = self.shower_norm(slots + self.slot_dropout(shower_context))
        slots = self.slot_ffn_norm(slots + self.slot_dropout(self.slot_ffn(slots)))

        hit_scores = self.hit_score_proj(encoded)
        slot_scores = self.slot_score_proj(slots)
        dynamic_logits = torch.einsum("bnd,bkd->bnk", hit_scores, slot_scores)
        dynamic_logits = dynamic_logits / math.sqrt(self.d_model)
        logits = self.base_head(encoded) + dynamic_logits + self.class_bias
        return logits.squeeze(0) if single_event else logits
