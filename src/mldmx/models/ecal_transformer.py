import torch
import torch.nn as nn


class ECalHitTransformer(nn.Module):
    """
    Per-hit ECal origin classifier using self-attention over one event.

    No explicit positional encoding is used; hit coordinates are already part of
    the input feature tensor.
    """

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
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
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
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ECal hit features with shape [N_hits, F].

        Returns:
            Per-hit logits with shape [N_hits, out_dim].
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [N_hits, F], got {tuple(x.shape)}")

        x = self.input_proj(x).unsqueeze(0)
        x = self.encoder(x).squeeze(0)
        return self.head(x)
