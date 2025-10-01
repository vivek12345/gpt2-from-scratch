"""RMS normalization (alternative to layer normalization)."""

import torch 
import torch.nn as nn


class RMSNorm(nn.Module):
	"""
	Root Mean Square normalization.
	Simpler alternative to LayerNorm - only uses RMS instead of mean and variance.
	"""
	
	def __init__(self, emb_dim, eps=1e-5):
		super().__init__()
		self.eps = eps
		self.scale = nn.Parameter(torch.ones(emb_dim))

	def forward(self, x):
		"""
		This is much simpler way to normalize. we just take the mean of square of values and then sqrt it to get rms value
		input = [2, 4, 6, 8, 10] 
		rms = sqrt(( 4 + 16 + 36 + 64 + 100) / 5)
		rms = sqrt(( 4 + 16 + 36 + 64 + 100) / 5)
		rms = sqrt(220 / 5)
		rms = sqrt(44)
		rms = 6.63

		so now normalized values = 2/6.63, 4/6.63, 6/6.63, 8/6.63, 10/6.63
								 = 0.301, 0.603, 0.904, 1.206, 1.508

		"""
		rms = torch.sqrt(torch.mean(x ** 2) + self.eps)
		normalized = x / rms

		return self.scale * normalized 