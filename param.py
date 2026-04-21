"""Project hyperparameter configuration.

Keeping these defaults in one place makes experiments easier to read,
reproduce, and modify.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
	seed: int = 42
	hidden_channels: int = 64
	out_channels: int = 64
	dropout: float = 0.2
	heads: int = 4
	learning_rate: float = 1e-3
	weight_decay: float = 1e-5
	epochs: int = 100
	k_neighbors: int = 10
	test_ratio: float = 0.1
	neg_sampling_ratio: float = 1.0