"""MLPF-inspired contributor-set and fraction model for ECal/TPad events."""

import torch
import torch.nn as nn


class ECalTpadContributorSetSlotModel(nn.Module):
    """Predict event-level electron-slot validity and per-hit contributor sets.

    This model intentionally lives alongside :class:`ECalTpadSlotModel`.  It
    has no independent event-count or hard-origin head.  Count, dominant
    origin, and mixed-hit decisions are derived from slot validity,
    contributor-set probabilities, and fraction predictions by the companion
    postprocessor.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 8,
        max_electrons: int = 3,
        min_electrons: int = 2,
        dropout: float = 0.1,
        use_type_embedding: bool = True,
    ):
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim and hidden_dim must be positive.")
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if max_electrons <= 0:
            raise ValueError("max_electrons must be positive.")
        if min_electrons < 0 or min_electrons > max_electrons:
            raise ValueError("min_electrons must be in 0..max_electrons.")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_electrons = int(max_electrons)
        self.min_electrons = int(min_electrons)
        self.num_fraction_classes = self.max_electrons + 1
        self.num_support_classes = 1 << self.max_electrons
        self.use_type_embedding = bool(use_type_embedding)

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_embedding = nn.Embedding(2, hidden_dim) if use_type_embedding else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

        def token_head(out_dim: int):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        self.support_head = token_head(self.num_support_classes)
        self.fraction_head = token_head(self.num_fraction_classes)
        self.event_summary = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_valid_head = token_head(self.max_electrons)

    def _infer_node_type(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] >= 2:
            return (x[..., 1] > x[..., 0]).to(dtype=torch.long)
        return torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor | None = None,
        ecal_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return raw parallel heads for one event or a padded event batch."""
        if x.ndim not in (2, 3) or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected [N, {self.in_dim}] or [B, N, {self.in_dim}], got {tuple(x.shape)}."
            )
        if x.shape[-2] == 0:
            raise ValueError("Expected at least one event token.")

        single_event = x.ndim == 2
        if single_event:
            x = x.unsqueeze(0)
            if node_type is not None and node_type.ndim == 1:
                node_type = node_type.unsqueeze(0)
            if ecal_mask is not None and ecal_mask.ndim == 1:
                ecal_mask = ecal_mask.unsqueeze(0)
            if key_padding_mask is not None and key_padding_mask.ndim == 1:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        token_shape = x.shape[:2]
        for name, value in (("node_type", node_type), ("ecal_mask", ecal_mask)):
            if value is not None and value.shape != token_shape:
                raise ValueError(
                    f"Expected {name} with shape {tuple(token_shape)}, got {tuple(value.shape)}."
                )
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(token_shape, dtype=torch.bool, device=x.device)
        elif key_padding_mask.shape != token_shape:
            raise ValueError(
                f"Expected key_padding_mask with shape {tuple(token_shape)}, "
                f"got {tuple(key_padding_mask.shape)}."
            )
        else:
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)
        valid_mask = ~key_padding_mask
        if not bool(valid_mask.any(dim=1).all().item()):
            raise ValueError("Every event must contain at least one non-padding token.")

        hidden = self.input_proj(x)
        if self.type_embedding is not None:
            if node_type is None:
                node_type = self._infer_node_type(x)
            hidden = hidden + self.type_embedding(
                node_type.to(device=x.device, dtype=torch.long).clamp(0, 1)
            )

        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        valid_float = valid_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        num_tokens = valid_float.sum(dim=1).clamp_min(1.0)
        mean_repr = (encoded * valid_float).sum(dim=1) / num_tokens
        max_repr = encoded.masked_fill(~valid_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        event_repr = self.event_summary(
            torch.cat([mean_repr, max_repr, torch.log1p(num_tokens)], dim=-1)
        )

        fraction_logits = self.fraction_head(encoded)
        outputs = {
            "support_logits": self.support_head(encoded),
            "fraction_logits": fraction_logits,
            "raw_fraction_pred": torch.softmax(fraction_logits, dim=-1),
            "slot_valid_logits": self.slot_valid_head(event_repr),
        }
        if single_event:
            return {key: value.squeeze(0) for key, value in outputs.items()}
        return outputs
