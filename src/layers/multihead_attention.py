"""Multi-head attention mechanism for GPT-2."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
	"""Multi-head self-attention with causal masking."""
	
	def __init__(self, emb_dim, context_length, qkv_bias, dropout, num_heads):
		super().__init__()
		assert emb_dim % num_heads == 0, "d_out must be divisible by num_heads"
		self.d_out = emb_dim
		self.num_heads = num_heads
		self.head_dim = emb_dim // num_heads
		self.W_Query = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
		self.W_Key = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
		self.W_Value = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

		self.dropout = nn.Dropout(dropout)
		# Optional out project via linear layer
		self.out_project = nn.Linear(emb_dim, emb_dim)
		self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

	def forward(self, x):
		batch_size, context_length, emb_dim = x.shape
		queries = self.W_Query(x)
		keys = self.W_Key(x)
		values = self.W_Value(x)

		"""
		Original Wq*X
		[
			[
			  [ 0.0374,  0.5449,  0.6522,  0.1421],    ← Token 0 The
			  [ 1.7467, -1.1825, -0.8515, -0.0642],    ← Token 1 Weather
			  [ 1.0126, -0.5027, -0.4863, -0.3160],    ← Token 2 is 
			  [ 0.8023,  0.3643, -1.5382, -0.0089]     ← Token 3 bad
			]    
		]
		Shape (1,4,4)
		"""

		"""
		After splitting into multi heads (2)

		        Head 1: Emotion       Head 2: Grammar Check
		                   
		[
		  [ 
		     [ [ 0.0374,  0.5449],   [ 0.6522,  0.1421] ],    ← Token 0: The
		     [ [ 1.7467, -1.1825],   [-0.8515, -0.0642] ],    ← Token 1: Weather
		     [ [ 1.0126, -0.5027],   [-0.4863, -0.3160] ],    ← Token 2: is
		     [ [ 0.8023,  0.3643],   [-1.5382, -0.0089] ]     ← Token 3: bad
		  ]    
		]


		Shape (1,4,2, 2)
		"""

		# Split into multiple heads
		queries = queries.view(batch_size, context_length, self.num_heads, self.head_dim)
		keys = keys.view(batch_size, context_length, self.num_heads, self.head_dim)
		values = values.view(batch_size, context_length, self.num_heads, self.head_dim)

		# Transpose to get dimensions [batch_size, num_heads, context_length, head_dim]
		queries = queries.transpose(1,2)
		keys = keys.transpose(1,2)
		values = values.transpose(1,2)

		# Calculate attention scores
		attention_scores = queries @ keys.transpose(2,3)

		# Apply causal masking
		attention_scores.masked_fill_(self.mask.bool()[:context_length, :context_length], -torch.inf)

		attention_weights = torch.softmax(attention_scores, dim=-1)
		attention_weights = self.dropout(attention_weights)

		"""
		After multiplying with values and transpose 1,2 again
		Shape: [1, 4, 2, 2]
		[
		  [ 
		    [ [v00_0, v00_1], [v10_0, v10_1] ],    ← Token 0 The 
		    [ [v01_0, v01_1], [v11_0, v11_1] ],    ← Token 1 Weather
		    [ [v02_0, v02_1], [v12_0, v12_1] ],    ← Token 2 is 
		    [ [v03_0, v03_1], [v13_0, v13_1] ]     ← Token 3 bad
		  ]    
		]
                 ↑ Head 0              ↑ Head 1
        """

		# Compute context vectors and merge heads
		context_vectors = (attention_weights @ values).transpose(1,2)
		"""
		print("context_vectors shape", context_vectors.shape) 
		"""
		
		"""
		Finally we merge
		Shape: [1, 4, 4]
		[
		  [ 
		  	[v00_0, v00_1, v10_0, v10_1],    ← Token 0 The
		    [v01_0, v01_1, v11_0, v11_1],    ← Token 1 Weather
		    [v02_0, v02_1, v12_0, v12_1],    ← Token 2 is 
		    [v03_0, v03_1, v13_0, v13_1]     ← Token 3 bad
		  ]
		]
		"""
		context_vectors = context_vectors.contiguous().view(batch_size, context_length, self.d_out)

		context_vectors = self.out_project(context_vectors)

		return context_vectors 