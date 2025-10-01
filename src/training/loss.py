"""Loss calculation functions for GPT-2 training."""

import torch


def calculate_cross_entropy_loss(input_batch, target_batch, model, device):
	"""
	Calculate cross-entropy loss for a single batch.
	
	The loss measures how well the model's predictions match the target tokens.
	
	Args:
		input_batch: Input token IDs [batch_size, seq_len]
		target_batch: Target token IDs [batch_size, seq_len]
		model: GPT model
		device: Device to run on (cpu/cuda/mps)
	
	Returns:
		Loss value (scalar tensor)
	"""
	# Transfer batch to device
	input_batch = input_batch.to(device)
	target_batch = target_batch.to(device)

	# Get model predictions
	logits = model(input_batch)

	# Calculate cross-entropy loss
	# Flatten from [batch_size, seq_len, vocab_size] to [batch_size*seq_len, vocab_size]
	"""

	
	The way this works internally is that 
	A) we first find out the real probablility prediction of our target word(correct word) inside the output for every batch
	for example for first batch it is 
	target_probs_1 = torch.tensor([
    	probas[0, 0, target[0][0]], find the probablity in batch 0 in first row and the probablity of the word weather for exmaple
	    probas[0, 1, target[0][1]],
	    probas[0, 2, target[0][2]],
	    probas[0, 3, target[0][3]],
	])
	This is done via 
	text_idx = 0
	target_probs_1 = probas[text_idx, [0,1,2,3], target[text_idx]]
	similarly for batch 1, it will be 
	text_idx = 1
	target_probs_1 = probas[text_idx, [0,1,2,3], target[text_idx]]

	B) We then concatenate them
	C) Then we apply log to them  
		logs = tensor([-11.0854, -10.4663, -11.1842, -10.6150, -11.5573, -10.9805, -11.2694, -10.7761])
	D) Finally we multiply the mean of them by -1 which is -negative mean log probablility also called cross entropy loss
	-1*torch.mean(logs)

	All of this is done by the pytorch function
	"""
	
	loss = torch.nn.functional.cross_entropy(
		logits.flatten(0, 1), 
		target_batch.flatten()
	)

	return loss


def calculate_cross_entropy_loss_loader(data_loader, model, device, num_batches):
	"""
	Calculate average cross-entropy loss over multiple batches.
	
	Args:
		data_loader: DataLoader with input/target batches
		model: GPT model
		device: Device to run on (cpu/cuda/mps)
		num_batches: Number of batches to evaluate (None for all)
	
	Returns:
		Average loss across batches
	"""
	total_loss = 0.
	if len(data_loader) == 0:
		return float('nan')

	if num_batches is None:
		num_batches = len(data_loader)
	else:
		num_batches = min(len(data_loader), num_batches)

	for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
		if batch_idx >= num_batches:
			break 

		total_loss += calculate_cross_entropy_loss(input_batch, target_batch, model, device)

	return total_loss / num_batches 