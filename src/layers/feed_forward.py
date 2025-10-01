"""Feed-forward network for transformer blocks."""

import torch.nn as nn


class FeedForwardLayer(nn.Module):
	"""Position-wise feed-forward network with GELU activation."""
	
	def __init__(self, cfg):
		super().__init__()
		"""
			┌────────────┐
	        │   Output   │  ← emb_dim (e.g., 4 neurons)
	        │   ●  ●     │
	        │   ●  ●     │
	        └────┬───────┘
	             ▲
	             │
	       Linear Layer 2
	             ▲
	             │
	        GELU Activation
	             ▲
	             │
	     ┌───────────────────┐
	     │     Expansion     │  ← emb_dim * 4 (e.g., 16 neurons)
	     │  ● ● ● ● ● ● ● ●  │
	     │  ● ● ● ● ● ● ● ●  │
	     └────────┬──────────┘
				  ▲
	              │
	     	Linear Layer 1
	              ▲
	              │
	        ┌────────────┐
	        │   Input    │  ← emb_dim (e.g., 4 neurons)
	        │   ●  ●     │
	        │   ●  ●     │
	        └────────────┘

		"""
		self.layers = nn.Sequential(
			nn.Linear(cfg["emb_dim"], cfg["emb_dim"]*4),
			nn.GELU(),
			nn.Linear(cfg["emb_dim"]*4, cfg["emb_dim"]),
		)

	def forward(self, x):
		return self.layers(x) 