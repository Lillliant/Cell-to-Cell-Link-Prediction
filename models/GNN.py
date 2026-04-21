"""Baseline GNN encoder.

This module provides a simple GraphSAGE encoder that can be used as a
baseline model for node representation learning in link prediction.
"""

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GNNEncoder(nn.Module):
	"""Two-layer GraphSAGE encoder used as a baseline GNN."""

	def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.2):
		super().__init__()
		self.conv1 = SAGEConv(in_channels, hidden_channels)
		self.conv2 = SAGEConv(hidden_channels, out_channels)
		self.dropout = dropout

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, hyperedge_index: torch.Tensor | None = None) -> torch.Tensor:
		# hyperedge_index is accepted for API compatibility with other models.
		del hyperedge_index
		x = self.conv1(x, edge_index)
		x = torch.relu(x)
		x = nn.functional.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, edge_index)
		return x
