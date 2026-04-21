"""Project hyperparameter configuration.

Keeping these defaults in one place makes experiments easier to read,
reproduce, and modify.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
	seed: int = 42
	hidden_channels: int = 64
	out_channels: int = 64 # Final node embedding size used by the edge decoder dot product.
	dropout: float = 0.2 # Fraction of hidden units randomly dropped during training regularization.
	heads: int = 4 # Number of attention heads (used by GAT/Graph Transformer only).
	learning_rate: float = 1e-3
	weight_decay: float = 1e-5 # L2 regularization strength to reduce overfitting on training edges.
	epochs: int = 100
	k_neighbors: int = 10 # k in kNN graph building for h5ad datasets (connect each node to k neighbors).
	val_ratio: float = 0.1 # Fraction of edges reserved for validation during RandomLinkSplit.
	test_ratio: float = 0.1
	neg_sampling_ratio: float = 1.0 # Negative:positive sample ratio for link prediction labels (1.0 = balanced).