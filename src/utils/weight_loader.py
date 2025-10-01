"""Utilities for loading pretrained OpenAI GPT-2 weights into PyTorch model."""

import torch
import numpy as np


def assign(left, right):
	"""Assign numpy array to PyTorch parameter with shape validation."""
	if left.shape != right.shape:
		raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
	return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
	"""
	Load pretrained GPT-2 weights from OpenAI checkpoint into PyTorch model.
	
	This function maps the parameter names and shapes from the TensorFlow checkpoint
	to our PyTorch model architecture.
	
	Args:
		gpt: GPTModel instance
		params: Dictionary of parameters loaded from TensorFlow checkpoint
	
	Note: The model must have the following structure:
		- token_embedding_layer (tok_emb)
		- positional_embedding_layer (pos_emb)
		- blocks (trf_blocks) with:
			- attn (att) with W_Query, W_Key, W_Value, out_project
			- ffl (ff) with layers
			- layer1 (norm1) and layer2 (norm2)
		- final_layer_norm (final_norm)
		- output (out_head)
	"""
	# Load token and positional embeddings
	gpt.positional_embedding_layer.weight = assign(
		gpt.positional_embedding_layer.weight, params["wpe"]
	)
	gpt.token_embedding_layer.weight = assign(
		gpt.token_embedding_layer.weight, params["wte"]
	)

	# Load weights for each transformer block
	for b in range(len(params["blocks"])):
		# Split combined QKV weights
		q_w, k_w, v_w = np.split(
			(params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
		)
		gpt.blocks[b].attn.W_Query.weight = assign(
			gpt.blocks[b].attn.W_Query.weight, q_w.T
		)
		gpt.blocks[b].attn.W_Key.weight = assign(
			gpt.blocks[b].attn.W_Key.weight, k_w.T
		)
		gpt.blocks[b].attn.W_Value.weight = assign(
			gpt.blocks[b].attn.W_Value.weight, v_w.T
		)

		# Split combined QKV biases
		q_b, k_b, v_b = np.split(
			(params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
		)
		gpt.blocks[b].attn.W_Query.bias = assign(
			gpt.blocks[b].attn.W_Query.bias, q_b
		)
		gpt.blocks[b].attn.W_Key.bias = assign(
			gpt.blocks[b].attn.W_Key.bias, k_b
		)
		gpt.blocks[b].attn.W_Value.bias = assign(
			gpt.blocks[b].attn.W_Value.bias, v_b
		)

		# Load attention output projection
		gpt.blocks[b].attn.out_project.weight = assign(
			gpt.blocks[b].attn.out_project.weight,
			params["blocks"][b]["attn"]["c_proj"]["w"].T
		)
		gpt.blocks[b].attn.out_project.bias = assign(
			gpt.blocks[b].attn.out_project.bias,
			params["blocks"][b]["attn"]["c_proj"]["b"]
		)

		# Load feed-forward network weights
		gpt.blocks[b].ffl.layers[0].weight = assign(
			gpt.blocks[b].ffl.layers[0].weight,
			params["blocks"][b]["mlp"]["c_fc"]["w"].T
		)
		gpt.blocks[b].ffl.layers[0].bias = assign(
			gpt.blocks[b].ffl.layers[0].bias,
			params["blocks"][b]["mlp"]["c_fc"]["b"]
		)
		gpt.blocks[b].ffl.layers[2].weight = assign(
			gpt.blocks[b].ffl.layers[2].weight,
			params["blocks"][b]["mlp"]["c_proj"]["w"].T
		)
		gpt.blocks[b].ffl.layers[2].bias = assign(
			gpt.blocks[b].ffl.layers[2].bias,
			params["blocks"][b]["mlp"]["c_proj"]["b"]
		)

		# Load layer normalization parameters
		gpt.blocks[b].layer1.scale = assign(
			gpt.blocks[b].layer1.scale,
			params["blocks"][b]["ln_1"]["g"]
		)
		gpt.blocks[b].layer1.shift = assign(
			gpt.blocks[b].layer1.shift,
			params["blocks"][b]["ln_1"]["b"]
		)
		gpt.blocks[b].layer2.scale = assign(
			gpt.blocks[b].layer2.scale,
			params["blocks"][b]["ln_2"]["g"]
		)
		gpt.blocks[b].layer2.shift = assign(
			gpt.blocks[b].layer2.shift,
			params["blocks"][b]["ln_2"]["b"]
		)

	# Load final layer normalization
	gpt.final_layer_norm.scale = assign(
		gpt.final_layer_norm.scale, params["g"]
	)
	gpt.final_layer_norm.shift = assign(
		gpt.final_layer_norm.shift, params["b"]
	)
	
	# Load output head (shares weights with token embeddings)
	gpt.output.weight = assign(gpt.output.weight, params["wte"]) 