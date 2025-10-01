"""Causal self-attention mechanism """

import torch
import torch.nn as nn


class CausalAttention(nn.Module):
	"""
	Causal self-attention with masking to prevent attending to future tokens.
	"""
	
	def __init__(self, din, dout, dropout, context_length, qkv_bias=False):
		super().__init__()
		self.W_Query = nn.Linear(din, dout, bias=qkv_bias)
		self.W_Key = nn.Linear(din, dout, bias=qkv_bias)
		self.W_Value = nn.Linear(din, dout, bias=qkv_bias)
		# this is to randomly drop some more of tokens during training to avoid overfitting
		self.dropout_layer = nn.Dropout(dropout)

		# we do this to make sure this mask can be moved to the device if gpu or cpu
		# the mask is basically a upper traingular identity matrix with above diagonal all 1
		self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

	def forward(self, x):
		batch_size, context_length, emb_dim = x.shape

		queries = self.W_Query(x)
		keys = self.W_Key(x)
		values = self.W_Value(x)

		attention_scores = queries @ keys.transpose(1,2)

		""" Masked Attention (Causal Masking)

			Let context_length = 4

			Step 1: Raw Attention Scores (Logits before softmax)
			------------------------------------------------------
			            Token_0   Token_1   Token_2   Token_3
			Token_0     0.9       1.1       1.3       0.7
			Token_1     1.0       0.8       1.2       1.5
			Token_2     0.6       1.4       0.5       1.0
			Token_3     0.3       0.9       1.1       0.4

			Step 2: Causal Mask (Upper Triangle Excl. Diagonal)
			---------------------------------------------------
			            T0   T1   T2   T3
			Token_0     0    1    1    1
			Token_1     0    0    1    1
			Token_2     0    0    0    1
			Token_3     0    0    0    0

			(1 = Masked, 0 = Allowed)

			Step 3: After Applying Mask (Replace 1s with -inf)
			---------------------------------------------------
			            Token_0   Token_1   Token_2   Token_3
			Token_0     0.9       -inf      -inf      -inf
			Token_1     1.0       0.8       -inf      -inf
			Token_2     0.6       1.4       0.5       -inf
			Token_3     0.3       0.9       1.1       0.4

			Step 4: After Softmax (Zero Prob for Masked Tokens)
			---------------------------------------------------
			            Token_0   Token_1   Token_2   Token_3
			Token_0     1.00      0.00      0.00      0.00
			Token_1     0.58      0.42      0.00      0.00
			Token_2     0.28      0.53      0.19      0.00
			Token_3     0.20      0.33      0.36      0.11



		here we convert the mask to bool which is True for 1 and false for 0 and then replace all attention weights where it is 1 to -INF
		
		This is to make sure all future tokens are masked. since we created a mask of upper triangula matrix so all future is masked
		"""

		# Apply causal masking: prevent attending to future tokens
		attention_scores.masked_fill_(self.mask.bool()[:context_length, :context_length], -torch.inf)

		attention_weights = torch.softmax(attention_scores, dim=1)
		attention_weights = self.dropout_layer(attention_weights)

		context_vectors = attention_weights @ values

		return context_vectors 