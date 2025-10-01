"""Basic self-attention mechanism (for educational purposes)."""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
	"""
	Basic self-attention without causal masking.
	Note: This is a simpler version used for learning. The main model uses MultiHeadAttention.
	"""
	
	def __init__(self, din, dout, qkv_bias=False):
		super().__init__()
		self.W_Query = nn.Linear(din, dout, bias=qkv_bias)
		self.W_Key = nn.Linear(din, dout, bias=qkv_bias)
		self.W_Value = nn.Linear(din, dout, bias=qkv_bias)

	def forward(self, x):
		batch_size, context_length, emb_dim = x.shape

		queries = self.W_Query(x)
		keys = self.W_Key(x)
		values = self.W_Value(x)

		# Calculate attention scores
		attention_scores = queries @ keys.transpose(1,2)
		
		# Apply softmax to get attention weights
		attention_weights = torch.softmax(attention_scores, dim=1)

		""" finally our context vector or the new vector representation of every word includes info about how every word attends to the other
		For example if the text is The weather is bad
		Take the value information from all words, but weight it by how much attention each word deserves"

		This creates vectors that are context-aware - they contain information not just about the word itself, but about its relationships with other words in the sentence.

		So after this I would know that the word The is taking about weather or bad is taking about weather being bad. so now the feature of the vectors are more informed
		"""
		# Create context-aware vectors
		context_vectors = attention_weights @ values

		return context_vectors 