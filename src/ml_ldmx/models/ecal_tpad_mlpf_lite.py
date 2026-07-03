import torch
import torch.nn as nn


class ECalTpadMLPFLiteTransformer(nn.Module):
    """
    Shared event-level transformer encoder with ECal-origin multi-task heads.

    The model accepts the combined ECal + TriggerPadTracks node features used by
    the current context-token prototypes and returns per-node outputs. Training
    scripts should apply losses only on ECal nodes.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        out_dim: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.origin_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )
        self.fraction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x_combined: torch.Tensor):
        """
        Args:
            x_combined: Combined event-node features with shape [N_total, F].

        Returns:
            Dict containing per-node origin logits, fraction logits, and
            softmax-normalized fraction predictions, each with shape [N_total, 3].
        """
        if x_combined.ndim != 2:
            raise ValueError(
                f"Expected x_combined with shape [N_total, F], got {tuple(x_combined.shape)}"
            )

        h = self.input_proj(x_combined).unsqueeze(0)
        h = self.encoder(h).squeeze(0)
        fraction_logits = self.fraction_head(h)
        return {
            "origin_logits": self.origin_head(h),
            "fraction_logits": fraction_logits,
            "fraction_pred": torch.softmax(fraction_logits, dim=-1),
        }
