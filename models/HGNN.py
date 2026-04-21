"""Hypergraph encoder for link prediction.

This implementation uses PyG's HypergraphConv. The training pipeline is
expected to pass a hyperedge incidence index in addition to edge_index.
"""

import torch
from torch import nn
from torch_geometric.nn import HypergraphConv


class HGNNEncoder(nn.Module):
	"""Two-layer hypergraph neural network encoder."""

	def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.2):
		super().__init__()
		self.conv1 = HypergraphConv(in_channels, hidden_channels)
		self.conv2 = HypergraphConv(hidden_channels, out_channels)
		self.dropout = dropout

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, hyperedge_index: torch.Tensor | None = None) -> torch.Tensor:
		del edge_index
		if hyperedge_index is None:
			raise ValueError("HGNNEncoder requires hyperedge_index.")
		# Message passing happens through node-hyperedge incidence relationships.
		x = self.conv1(x, hyperedge_index)
		x = torch.relu(x)
		x = nn.functional.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, hyperedge_index)
		return x
