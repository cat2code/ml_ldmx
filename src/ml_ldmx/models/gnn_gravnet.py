"""Maintained GravNetConv hit-origin classification baselines."""

import torch
import torch.nn as nn
from torch_geometric.nn import GravNetConv


class _GravNetHitClassifier(nn.Module):
    """Per-node classifier whose neighborhoods are learned by GravNetConv."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 3,
        num_layers: int = 2,
        space_dimensions: int = 4,
        propagate_dimensions: int = 32,
        k: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim, hidden_dim, and out_dim must be positive.")
        if num_layers <= 0 or space_dimensions <= 0 or propagate_dimensions <= 0 or k <= 0:
            raise ValueError("num_layers, space_dimensions, propagate_dimensions, and k must be positive.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList(
            [
                GravNetConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    space_dimensions=space_dimensions,
                    propagate_dimensions=propagate_dimensions,
                    k=k,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """
        Return per-input-node origin-class logits.

        Args:
            x: Node features with shape ``[N_nodes, in_dim]``.
            batch: Optional graph-assignment vector with shape ``[N_nodes]``.
                Passing ``None`` represents one event.
        """
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x with shape [N_nodes, {self.in_dim}], got {tuple(x.shape)}.")
        if x.shape[0] == 0:
            raise ValueError("Expected at least one node, got an empty tensor.")
        if batch is not None and (batch.ndim != 1 or batch.shape[0] != x.shape[0]):
            raise ValueError(f"Expected batch with shape [{x.shape[0]}], got {tuple(batch.shape)}.")

        h = self.input_proj(x)
        for conv in self.convs:
            h = h + self.dropout(torch.relu(conv(h, batch=batch)))
        return self.head(h)


class ECalGravNet(_GravNetHitClassifier):
    """ECal-only GravNetConv hit-origin classifier."""


class ECalTpadGravNet(_GravNetHitClassifier):
    """ECal plus TriggerPadTracks GravNetConv hit-origin classifier."""
