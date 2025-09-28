import torch
import torch.nn as nn

class SelfAttention(nn.Module):
	def __init__(self, din, dout, qkv_bias=False):
		super().__init__()
		# initialize weights for Q, K and V which are trainable parameter. Linear layer makes it trainable with bias
		# we could have also done nn.Parameter()
		self.W_Query = nn.Linear(din, dout, bias=qkv_bias)
		self.W_Key = nn.Linear(din, dout, bias=qkv_bias)
		self.W_Value = nn.Linear(din, dout, bias=qkv_bias)

	def forward(self, x):
		# our input is of the shape [2, 4, 768]
		batch_size, context_length, emb_dim = x.shape

		# get the queries which X * W_Q
		queries = self.W_Query(x)
		# get the keys which X * W_K
		keys = self.W_Key(x)

		# get the values which X * W_V
		values = self.W_Value(x)

		# see how much every word attends to the other by checking Query of every word against keys of others
		attention_scores = queries @ keys.transpose(1,2)


		# finally apply softmax to keep probability distribution 0 and 1
		attention_weights = torch.softmax(attention_scores, dim=1)

		""" finally our context vector or the new vector representation of every word includes info about how every word attends to the other
		For example if the text is The weather is bad
		Take the value information from all words, but weight it by how much attention each word deserves"

		This creates vectors that are context-aware - they contain information not just about the word itself, but about its relationships with other words in the sentence.

		So after this I would know that the word The is taking about weather or bad is taking about weather being bad. so now the feature of the vectors are more informed
		"""

		context_vectors = attention_weights @ values

		return context_vectors

def main():
	text = "The weather is bad"
	token_ids = [0,1,2,3]

	vocab = {
		0: "The",
		1: "Weather",
		2: "is",
		3: "bad"
	}

	config = {
		"vocab_size": 4,
		"emb_dim": 3,
		"context_length": 4
	}

	emb_layer = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
	positional_layer = torch.nn.Embedding(config["context_length"], config["emb_dim"])

	token_embeddings = emb_layer(torch.tensor(token_ids))
	positional_embeddings = positional_layer(torch.arange(config["context_length"]))

	input_embeddings = token_embeddings + positional_embeddings

	self_v1 = SelfAttention(config["emb_dim"], 2, qkv_bias=True)
	context_vectors = self_v1(input_embeddings.unsqueeze(0))
	print(context_vectors)


