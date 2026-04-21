"""GCN encoder for link prediction."""

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
	"""Two-layer GCN encoder."""

	def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.2):
		super().__init__()
		self.conv1 = GCNConv(in_channels, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, out_channels)
		self.dropout = dropout

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, hyperedge_index: torch.Tensor | None = None) -> torch.Tensor:
		del hyperedge_index
		# GCN layer with normalized neighbor aggregation.
		x = self.conv1(x, edge_index)
		x = torch.relu(x)
		x = nn.functional.dropout(x, p=self.dropout, training=self.training)
		# Final embedding layer for link prediction decoder.
		x = self.conv2(x, edge_index)
		return x
