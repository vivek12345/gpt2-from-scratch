#!/usr/bin/env python3
"""Demonstration of different attention mechanisms."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import tiktoken
from src.layers.self_attention import SelfAttention
from src.layers.causal_attention import CausalAttention
from src.layers.multihead_attention import MultiHeadAttention


def main():
	torch.manual_seed(123)
	
	# Setup
	text = "The weather is bad"
	tokenizer = tiktoken.get_encoding("gpt2")
	token_ids = tokenizer.encode(text)
	
	config = {
		"vocab_size": 50257,
		"emb_dim": 768,
		"context_length": len(token_ids),
		"dropout": 0.1
	}

	# Create embeddings
	emb_layer = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
	positional_layer = torch.nn.Embedding(config["context_length"], config["emb_dim"])
	
	token_embeddings = emb_layer(torch.tensor(token_ids))
	positional_embeddings = positional_layer(torch.arange(config["context_length"]))
	input_embeddings = (token_embeddings + positional_embeddings).unsqueeze(0)

	print("=" * 60)
	print("Attention Mechanisms Demo")
	print("=" * 60)
	print(f"Input text: {text}")
	print(f"Input shape: {input_embeddings.shape}")
	print()

	# 1. Self Attention (no masking)
	print("-" * 60)
	print("1. Self Attention (can attend to all tokens)")
	print("-" * 60)
	self_attn = SelfAttention(
		din=config["emb_dim"],
		dout=config["emb_dim"],
		qkv_bias=True
	)
	with torch.no_grad():
		output = self_attn(input_embeddings)
	print(f"Output shape: {output.shape}")
	print()

	# 2. Causal Attention (with masking)
	print("-" * 60)
	print("2. Causal Attention (can only attend to past tokens)")
	print("-" * 60)
	causal_attn = CausalAttention(
		din=config["emb_dim"],
		dout=config["emb_dim"],
		dropout=config["dropout"],
		context_length=config["context_length"],
		qkv_bias=True
	)
	with torch.no_grad():
		output = causal_attn(input_embeddings)
	print(f"Output shape: {output.shape}")
	print()

	# 3. Multi-Head Attention
	print("-" * 60)
	print("3. Multi-Head Attention (12 heads, causal)")
	print("-" * 60)
	multihead_attn = MultiHeadAttention(
		emb_dim=config["emb_dim"],
		context_length=config["context_length"],
		qkv_bias=True,
		dropout=config["dropout"],
		num_heads=12
	)
	with torch.no_grad():
		output = multihead_attn(input_embeddings)
	print(f"Output shape: {output.shape}")
	print(f"Number of heads: 12")
	print(f"Head dimension: {config['emb_dim'] // 12}")
	print()

	print("=" * 60)
	print("All attention mechanisms transform embeddings while")
	print("preserving the shape [batch_size, seq_len, emb_dim]")
	print("=" * 60)


if __name__ == "__main__":
	main()
