"""Tokenization utilities for converting between text and token IDs."""

import torch


def text_to_token_ids(text, tokenizer):
	"""
	Convert text to token IDs.
	
	Args:
		text: Input text string
		tokenizer: Tokenizer object (e.g., tiktoken)
	
	Returns:
		Token IDs as a batched tensor [1, seq_len]
	"""
	# Get token ids as a list [464, 6193, 318, 2089]
	encoded = tokenizer.encode(text)

	# Convert to tensor: tensor([464, 6193, 318, 2089])
	token_ids = torch.tensor(encoded)

	# Add batch dimension: tensor([[464, 6193, 318, 2089]])
	return token_ids.unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
	"""
	Convert token IDs back to text.
	
	Args:
		token_ids: Batched tensor of token IDs [1, seq_len]
		tokenizer: Tokenizer object (e.g., tiktoken)
	
	Returns:
		Decoded text string
	"""
	# Remove batch dimension: tensor([464, 6193, 318, 2089])
	token_ids = token_ids.squeeze(0)

	# Convert to list for tokenizer: [464, 6193, 318, 2089]
	encoded = token_ids.tolist()

	# Decode to text: "The weather is bad"
	decoded = tokenizer.decode(encoded)

	return decoded 