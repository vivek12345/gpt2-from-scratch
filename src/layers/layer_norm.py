"""Layer normalization for stabilizing training."""

import torch 
import torch.nn as nn


class LayerNorm(nn.Module):
	"""Layer normalization with learnable scale and shift parameters."""
	
	def __init__(self, emb_dim, eps=1e-5):
		super().__init__()
		self.eps = eps
		self.shift = nn.Parameter(torch.zeros(emb_dim))
		self.scale = nn.Parameter(torch.ones(emb_dim))

	def forward(self, x):
		"""
		We first calculate the mean and variance
		let's say for simple example:
		input = [2, 4, 6, 8, 10] 
		mean = (2 + 4 + 6 + 8 + 10)/5 = 6
		variance = (sqr(2-6) + sqr(4-6) + sqr(6-6) + sqr(8-6) + sqr(10-6)) / 5 
				 = (16 + 4 + 0 + 4 + 16) / 5
				 = 8
		S.D = Sqrt(variance) = 4

		Normalized values = (2-6)/Sqrt(variance), (4-6)Sqrt(variance), (6-6)Sqrt(variance), (8-6)Sqrt(variance), (10-6)Sqrt(variance)
						  = -1, -0.5, 0, 0.5, 1

		same thing we do with tensors. except for 3 changes. 
		1) we add self.eps to avoid divide by zero error
		2) we add trainable param self.scale and self.shift
		3) self.scale is 1 and self.shift is 0. so no change in the normalized value
		4) we adjust scale or shift to make sure the output is not always the same centered around 0. it is normalized but still produces different output
		"""
		mean = torch.mean(x, dim=-1, keepdim=True)
		variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)

		normalized = (x - mean) / torch.sqrt(variance + self.eps)

		return self.scale * normalized + self.shift 