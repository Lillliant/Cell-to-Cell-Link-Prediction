"""GAT encoder for link prediction."""

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
	"""Two-layer graph attention encoder."""

	def __init__(
		self,
		in_channels: int,
		hidden_channels: int,
		out_channels: int,
		dropout: float = 0.2,
		heads: int = 4,
	):
		super().__init__()
		self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
		self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
		self.dropout = dropout

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, hyperedge_index: torch.Tensor | None = None) -> torch.Tensor:
		del hyperedge_index
		x = self.conv1(x, edge_index)
		x = torch.relu(x)
		x = nn.functional.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, edge_index)
		return x
