"""Data loading utilities for GPT-2 training."""

from torch.utils.data import DataLoader, Dataset
import torch


class GPT2Dataset(Dataset):
	"""Dataset for GPT-2 training with sliding window approach."""
	
	def __init__(self, text, tokenizer, context_length, stride):
		self.input_ids = []
		self.target_ids = []

		token_ids = tokenizer.encode(text)

		for i in range(0, len(token_ids)-context_length, stride):
			input_chunk = token_ids[i:i+context_length]
			target_chunk = token_ids[i+1:i+context_length+1]
			self.input_ids.append(torch.tensor(input_chunk))
			self.target_ids.append(torch.tensor(target_chunk))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.target_ids[idx]


def create_data_loader(text, tokenizer, context_length, stride, batch_size, 
                       shuffle=True, drop_last=True, num_workers=0):
	"""Create a PyTorch DataLoader for GPT-2 training."""
	dataset = GPT2Dataset(text, tokenizer, context_length, stride)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		drop_last=drop_last,
		num_workers=num_workers
	)

	return dataloader 