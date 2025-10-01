#!/usr/bin/env python3
"""Example of loading pretrained OpenAI GPT-2 weights and generating text."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tiktoken
import torch
from src.models.gpt2 import GPTModel
from src.utils.text_generation import generate_text_simple
from src.data.tokenization import token_ids_to_text
from src.utils.weight_loader import load_weights_into_gpt

# Note: You need to run scripts/download_gpt2_weights.py first
from scripts.download_gpt2_weights import download_and_load_gpt2


def main():
	"""Load pretrained GPT-2 weights and generate text."""
	print("=" * 70)
	print("Loading Pretrained GPT-2 Weights Example")
	print("=" * 70)
	print()
	
	# Step 1: Download pretrained weights (if not already downloaded)
	print("Step 1: Downloading pretrained weights...")
	models_dir = os.path.join(os.path.dirname(__file__), '..', 'gpt2')
	settings, params = download_and_load_gpt2(model_size="124M", models_dir=models_dir)
	print(f"Loaded settings: {settings}")
	print()
	
	# Step 2: Create model with matching configuration
	print("Step 2: Creating GPT-2 model...")
	GPT2_CONFIG = {
		"vocab_size": settings["n_vocab"],      # 50257
		"emb_dim": settings["n_embd"],          # 768
		"context_length": settings["n_ctx"],    # 1024
		"num_layers": settings["n_layer"],      # 12
		"num_heads": settings["n_head"],        # 12
		"qkv_bias": True,
		"dropout": 0.1
	}
	
	model = GPTModel(GPT2_CONFIG)
	print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
	print()
	
	# Step 3: Load pretrained weights into model
	print("Step 3: Loading pretrained weights into model...")
	load_weights_into_gpt(model, params)
	print("✓ Weights loaded successfully!")
	print()
	
	# Step 4: Set model to evaluation mode
	model.eval()
	tokenizer = tiktoken.get_encoding("gpt2")
	
	# Step 5: Generate text
	print("=" * 70)
	print("Text Generation Examples")
	print("=" * 70)
	print()
	
	prompts = [
		"Once upon a time",
		"The meaning of life is",
		"Every effort moves you",
	]
	
	for prompt in prompts:
		print(f"Prompt: '{prompt}'")
		print("-" * 70)
		
		output = generate_text_simple(
			model=model,
			tokenizer=tokenizer,
			start_context=prompt,
			context_length=GPT2_CONFIG["context_length"],
			max_new_tokens=30,
			temperature=0.7,
			top_k=50
		)
		
		generated_text = token_ids_to_text(output, tokenizer)
		print(generated_text)
		print()
	
	print("=" * 70)
	print("Note: With pretrained weights, the model should generate")
	print("coherent and contextually appropriate text!")
	print("=" * 70)


if __name__ == "__main__":
	main() 