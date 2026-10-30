"""Temporal and graph-temporal forecasting models."""
import torch
from torch import nn


class GRUBaseline(nn.Module):
    def __init__(self, n_links, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_links, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_links)

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1])


class GraphGRU(nn.Module):
    def __init__(self, n_links, hidden=32):
        super().__init__()
        self.spatial = nn.Linear(1, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.n_links = n_links

    def forward(self, x, adjacency):
        # Aggregate neighboring link measurements at each input time step.
        z = torch.einsum("ij,btjf->btif", adjacency, x.unsqueeze(-1))
        z = torch.relu(self.spatial(z))
        b, t, n, h = z.shape
        z = z.permute(0, 2, 1, 3).reshape(b * n, t, h)
        _, state = self.gru(z)
        return self.head(state[-1]).reshape(b, n)
