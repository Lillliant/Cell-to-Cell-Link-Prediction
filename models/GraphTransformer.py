"""Graph Transformer encoder for link prediction."""

import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class GraphTransformerEncoder(nn.Module):
	"""Two-layer TransformerConv encoder."""

	def __init__(
		self,
		in_channels: int,
		hidden_channels: int,
		out_channels: int,
		dropout: float = 0.2,
		heads: int = 4,
	):
		super().__init__()
		# Transformer-style attention on graph neighborhoods with multiple heads.
		self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
		# Project back to a single embedding vector per node.
		self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
		self.dropout = dropout

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, hyperedge_index: torch.Tensor | None = None) -> torch.Tensor:
		del hyperedge_index
		x = self.conv1(x, edge_index)
		x = torch.relu(x)
		x = nn.functional.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, edge_index)
		return x
