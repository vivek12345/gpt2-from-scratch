#!/usr/bin/env python3
"""
Script to download pretrained GPT-2 weights from OpenAI.

Copyright (c) Sebastian Raschka under Apache License 2.0
Source: https://github.com/rasbt/LLMs-from-scratch
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import urllib.request
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
	"""Download and load GPT-2 weights from OpenAI."""
	# Validate model size
	allowed_sizes = ("124M", "355M", "774M", "1558M")
	if model_size not in allowed_sizes:
		raise ValueError(f"Model size not in {allowed_sizes}")

	# Define paths
	model_dir = os.path.join(models_dir, model_size)
	base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
	backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
	filenames = [
		"checkpoint", "encoder.json", "hparams.json",
		"model.ckpt.data-00000-of-00001", "model.ckpt.index",
		"model.ckpt.meta", "vocab.bpe"
	]

	# Download files
	os.makedirs(model_dir, exist_ok=True)
	for filename in filenames:
		file_url = os.path.join(base_url, model_size, filename)
		backup_url = os.path.join(backup_base_url, model_size, filename)
		file_path = os.path.join(model_dir, filename)
		download_file(file_url, file_path, backup_url)

	# Load settings and params
	tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
	settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
	params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

	return settings, params


def download_file(url, destination, backup_url=None):
	"""Download a file with progress bar."""
	def _attempt_download(download_url):
		with urllib.request.urlopen(download_url) as response:
			file_size = int(response.headers.get("Content-Length", 0))

			# Check if file exists and has the same size
			if os.path.exists(destination):
				file_size_local = os.path.getsize(destination)
				if file_size == file_size_local:
					print(f"File already exists and is up-to-date: {destination}")
					return True

			block_size = 1024  # 1 Kilobyte

			progress_bar_description = os.path.basename(download_url)
			with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
				with open(destination, "wb") as file:
					while True:
						chunk = response.read(block_size)
						if not chunk:
							break
						file.write(chunk)
						progress_bar.update(len(chunk))
			return True

	try:
		if _attempt_download(url):
			return
	except (urllib.error.HTTPError, urllib.error.URLError):
		if backup_url is not None:
			print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
			try:
				if _attempt_download(backup_url):
					return
			except urllib.error.HTTPError:
				pass

		error_message = (
			f"Failed to download from both primary URL ({url})"
			f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
			"\nCheck your internet connection or the file availability.\n"
			"For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
		)
		print(error_message)
	except Exception as e:
		print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
	"""Load GPT-2 parameters from TensorFlow checkpoint."""
	params = {"blocks": [{} for _ in range(settings["n_layer"])]}

	for name, _ in tf.train.list_variables(ckpt_path):
		variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
		variable_name_parts = name.split("/")[1:]

		target_dict = params
		if variable_name_parts[0].startswith("h"):
			layer_number = int(variable_name_parts[0][1:])
			target_dict = params["blocks"][layer_number]

		for key in variable_name_parts[1:-1]:
			target_dict = target_dict.setdefault(key, {})

		last_key = variable_name_parts[-1]
		target_dict[last_key] = variable_array

	return params


def main():
	"""Download GPT-2 124M weights."""
	models_dir = os.path.join(os.path.dirname(__file__), '..', 'gpt2')
	
	print("Downloading GPT-2 124M weights...")
	settings, params = download_and_load_gpt2(model_size="124M", models_dir=models_dir)
	
	print("\nSettings:")
	print(settings)
	print("\nParameter keys:")
	print(params.keys())
	print("\nDownload complete!")


if __name__ == "__main__":
	main() 