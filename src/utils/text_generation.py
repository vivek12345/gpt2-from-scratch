"""Text generation utilities for GPT-2."""

import torch
from src.data.tokenization import text_to_token_ids


def generate_text_simple(model, start_context, context_length, max_new_tokens, 
                         tokenizer, temperature, top_k=None, eos=None):
	"""
	Generate text using the GPT-2 model.
	
	Args:
		model: GPT model
		start_context: Starting text prompt
		context_length: Maximum context length
		max_new_tokens: Number of new tokens to generate
		tokenizer: Tokenizer object
		temperature: Sampling temperature (0 = greedy, >1 = more random)
		top_k: Optional top-k sampling
		eos: Optional end-of-sequence token
	
	Returns:
		Generated token IDs as tensor
	"""
	idx = text_to_token_ids(start_context, tokenizer)

	for i in range(max_new_tokens):
		idx_cond = idx[:, -context_length:]
		with torch.no_grad():
			logits = model(idx_cond)

		logits = logits[:, -1, :]

		# Apply top-k sampling if specified
		if top_k is not None:
			top_logits, top_position = torch.topk(logits, top_k)
			logits = torch.where(
				condition=logits < top_logits[:, -1],
				input=torch.tensor(float('-inf')),
				other=logits
			)

		# Sample next token
		if temperature > 0.0:
			logits = logits / temperature
			probas = torch.softmax(logits, dim=-1)
			next_token_id = torch.multinomial(probas, num_samples=1)
		else:
			probas = torch.softmax(logits, dim=-1)
			next_token_id = torch.argmax(probas, dim=-1, keepdim=True)
		
		# Check for end-of-sequence
		if next_token_id == eos:
			break

		idx = torch.cat((idx, next_token_id), dim=1)

	return idx 