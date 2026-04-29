import torch.nn as nn
from torch_geometric.nn import GraphConv


class ECalTriggerPadGNN(nn.Module):
    """Small per-node classifier for ECal hits with TriggerPadTracks context nodes."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 3):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.head(x)
