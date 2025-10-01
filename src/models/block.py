import torch.nn as nn
from src.layers import MultiHeadAttention, LayerNorm, FeedForwardLayer


class TransformerBlock(nn.Module):
	"""Transformer block with multi-head attention and feed-forward layers."""
	
	def __init__(self, cfg):
		super().__init__()
		self.layer1 = LayerNorm(cfg["emb_dim"])
		self.attn = MultiHeadAttention(
			emb_dim=cfg["emb_dim"],
			context_length=cfg["context_length"],
			qkv_bias=cfg["dropout"],
			dropout=cfg["dropout"],
			num_heads=cfg["num_heads"]
		)
		self.layer2 = LayerNorm(cfg["emb_dim"])
		self.ffl = FeedForwardLayer(cfg)
		self.dropout = nn.Dropout(cfg["dropout"])

	def forward(self, x):
		shortcut = x
		x = self.layer1(x)
		x = self.attn(x)
		x = self.dropout(x)
		x = x + shortcut

		shortcut = x

		x = self.layer2(x)
		x = self.ffl(x)
		x = self.dropout(x)
		x = x + shortcut

		return x 