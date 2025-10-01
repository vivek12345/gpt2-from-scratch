#!/usr/bin/env python3
"""Script to train GPT-2 from scratch."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tiktoken
import torch
from src.data.data_loader import create_data_loader
from src.models.gpt2 import GPTModel
from src.training.train import train_model


def main():
	# Load training data
	data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'verdict.txt')
	with open(data_path, "r") as f:
		text = f.read()

	# Split into train/validation
	train_split = 0.9
	split_idx = int(len(text) * train_split)
	train_data = text[:split_idx]
	val_data = text[split_idx:]

	# GPT-2 configuration
	GPT2_CONFIG = {
		"vocab_size": 50257,
		"emb_dim": 768,
		"context_length": 256,
		"num_layers": 12,
		"num_heads": 12,
		"qkv_bias": True,
		"dropout": 0.1
	}

	# Initialize tokenizer
	tokenizer = tiktoken.get_encoding("gpt2")

	# Create data loaders
	train_loader = create_data_loader(
		train_data,
		batch_size=2,
		tokenizer=tokenizer,
		context_length=GPT2_CONFIG["context_length"],
		stride=GPT2_CONFIG["context_length"],
		drop_last=True,
		shuffle=True,
		num_workers=0
	)

	val_loader = create_data_loader(
		val_data,
		batch_size=2,
		tokenizer=tokenizer,
		context_length=GPT2_CONFIG["context_length"],
		stride=GPT2_CONFIG["context_length"],
		drop_last=True,
		shuffle=True,
		num_workers=0
	)

	# Initialize model
	device = "cpu"  # Change to "cuda" or "mps" if available
	model = GPTModel(GPT2_CONFIG)
	model = model.to(device)

	# Initialize optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

	# Training parameters
	num_epochs = 10

	print("Starting training...")
	print(f"Device: {device}")
	print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
	
	# Train the model
	train_loss, val_loss, _ = train_model(
		num_epochs=num_epochs,
		training_loader=train_loader,
		val_loader=val_loader,
		device=device,
		model=model,
		optimizer=optimizer,
		tokenizer=tokenizer,
		eval_freq=5,
		eval_iter=5,
		start_context="The weather is bad"
	)

	print("\nTraining complete!")


if __name__ == "__main__":
	main() 