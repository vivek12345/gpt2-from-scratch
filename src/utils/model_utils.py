"""Model utilities for saving and loading."""

import torch


def save_model(model, name="model.pth"):
	"""
	Save model weights to file.
	
	Args:
		model: GPT model
		name: Output filename
	"""
	torch.save(model.state_dict(), name)
	print(f"Model saved to {name}")


def load_model(model, name="model.pth"):
	"""
	Load model weights from file.
	
	Args:
		model: GPT model instance
		name: Input filename
	
	Returns:
		Model with loaded weights
	"""
	model.load_state_dict(torch.load(name))
	print(f"Model loaded from {name}")
	return model 