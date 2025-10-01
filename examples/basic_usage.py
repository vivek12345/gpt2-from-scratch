#!/usr/bin/env python3
"""Basic usage example of the GPT-2 model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tiktoken
import torch
from src.models.gpt2 import GPTModel
from src.utils.text_generation import generate_text_simple
from src.data.tokenization import token_ids_to_text


def main():
	# Set random seed for reproducibility
	torch.manual_seed(123)
	
	# GPT-2 configuration
	GPT2_CONFIG = {
		"vocab_size": 50257,
		"emb_dim": 768,
		"context_length": 1024,
		"num_layers": 12,
		"num_heads": 12,
		"qkv_bias": True,
		"dropout": 0.1
	}

	# Initialize model and tokenizer
	model = GPTModel(GPT2_CONFIG)
	tokenizer = tiktoken.get_encoding("gpt2")

	# Example 1: Get model output (logits)
	print("=" * 50)
	print("Example 1: Forward pass")
	print("=" * 50)
	
	input_text = "The weather is bad"
	input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
	
	with torch.no_grad():
		logits = model(input_ids)
	
	print(f"Input: {input_text}")
	print(f"Input shape: {input_ids.shape}")
	print(f"Output logits shape: {logits.shape}")
	print()

	# Example 2: Generate text
	print("=" * 50)
	print("Example 2: Text generation")
	print("=" * 50)
	
	start_context = "Once upon a time"
	
	output = generate_text_simple(
		model=model,
		tokenizer=tokenizer,
		start_context=start_context,
		context_length=GPT2_CONFIG["context_length"],
		max_new_tokens=10,
		temperature=1.0,
		top_k=5
	)
	
	generated_text = token_ids_to_text(output, tokenizer)
	
	print(f"Start: {start_context}")
	print(f"Generated: {generated_text}")
	print()

	# Example 3: Model statistics
	print("=" * 50)
	print("Example 3: Model statistics")
	print("=" * 50)
	
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	
	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")
	print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")


if __name__ == "__main__":
	main() 