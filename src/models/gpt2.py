import torch
import torch.nn as nn
from .block import TransformerBlock
from src.layers import LayerNorm


class GPTModel(nn.Module):
	"""GPT-2 Model implementation."""
	
	def __init__(self, cfg):
		super().__init__()
		self.token_embedding_layer = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
		self.positional_embedding_layer = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

		self.dropout = nn.Dropout(cfg["dropout"])

		self.blocks = nn.Sequential(*[
			TransformerBlock(cfg)
			for _ in range(cfg["num_layers"])
		])

		self.final_layer_norm = LayerNorm(cfg["emb_dim"])
		self.output = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

	def forward(self, idx):
		batch_size, context_length = idx.shape
		token_embeddings = self.token_embedding_layer(idx)
		positional_embeddings = self.positional_embedding_layer(torch.arange(context_length))

		x = token_embeddings + positional_embeddings
		x = self.dropout(x)
		x = self.blocks(x)

		x = self.final_layer_norm(x)
		logits = self.output(x)

		return logits 