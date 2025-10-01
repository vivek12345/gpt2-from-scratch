"""Training utilities for GPT-2."""

import torch
from .loss import calculate_cross_entropy_loss, calculate_cross_entropy_loss_loader
from src.data.tokenization import token_ids_to_text
from src.utils.text_generation import generate_text_simple


def train_model(num_epochs, training_loader, val_loader, device, model, optimizer, 
                tokenizer, eval_freq, eval_iter, start_context):
	"""
	Train the GPT-2 model.
	
	Args:
		num_epochs: Number of training epochs
		training_loader: Training data loader
		val_loader: Validation data loader
		device: Device to train on (cpu/cuda/mps)
		model: GPT model
		optimizer: Optimizer (e.g., AdamW)
		tokenizer: Tokenizer for text generation
		eval_freq: Evaluate every N steps
		eval_iter: Number of batches to use for evaluation
		start_context: Start text for sample generation
	
	Returns:
		Tuple of (training_losses, validation_losses, tokens_seen)
	"""
	training_loss, validation_loss, track_tokens_seen = [], [], []
	tokens_seen, global_step = 0, -1

	for epoch in range(num_epochs):
		model.train()
		for batch_idx, (input_batch, target_batch) in enumerate(training_loader):
			optimizer.zero_grad()
			loss = calculate_cross_entropy_loss(input_batch, target_batch, model, device)
			loss.backward()
			optimizer.step()

			tokens_seen += torch.numel(input_batch)
			global_step += 1

			if global_step % eval_freq == 0:
				train_loss, val_loss = evaluate_model(
					training_loader,
					val_loader,
					device=device,
					model=model,
					eval_iter=eval_iter 
				)
				training_loss.append(train_loss)
				validation_loss.append(val_loss)
				track_tokens_seen.append(tokens_seen)
				print(f"Epoch: {epoch+1}, Step {global_step:06d}, Train Loss {train_loss:.3f}, Val loss {val_loss:.3f}")

		print_sample_result_after_every_epoch(start_context, model, tokenizer)

	return training_loss, validation_loss, track_tokens_seen


def print_sample_result_after_every_epoch(start_context, model, tokenizer):
	"""Generate and print sample text after each epoch."""
	model.eval()
	context_length = model.positional_embedding_layer.weight.shape[0]

	with torch.no_grad():
		output = generate_text_simple(
			model=model, 
			tokenizer=tokenizer, 
			start_context=start_context,
			context_length=context_length, 
			max_new_tokens=10,
			temperature=1.0
		)
		print(token_ids_to_text(output, tokenizer))

	model.train()


def evaluate_model(train_loader, val_loader, device, model, eval_iter):
	"""
	Evaluate model on training and validation data.
	
	Returns:
		Tuple of (train_loss, val_loss)
	"""
	model.eval()
	with torch.no_grad():
		train_loss = calculate_cross_entropy_loss_loader(
			train_loader,
			device=device,
			model=model,
			num_batches=eval_iter
		)
		val_loss = calculate_cross_entropy_loss_loader(
			val_loader,
			device=device,
			model=model,
			num_batches=eval_iter
		)

	model.train()

	return train_loss, val_loss 