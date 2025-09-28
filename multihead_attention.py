import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
	def __init__(self, din, dout, context_length, qkv_bias, dropout, num_heads):
		super().__init__()
		assert dout % num_heads == 0, "d_out must be divisible by num_heads"
		self.d_out = dout
		self.num_heads = num_heads
		self.head_dim = dout // num_heads
		self.W_Query = nn.Linear(din, dout, bias=qkv_bias)
		self.W_Key = nn.Linear(din, dout, bias=qkv_bias)
		self.W_Value = nn.Linear(din, dout, bias=qkv_bias)

		self.dropout = nn.Dropout(dropout)
		self.out_project = nn.Linear(dout, dout)
		self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

	def forward(self, x):
		batch_size, context_length, d_in = x.shape
		queries = self.W_Query(x)
		keys = self.W_Key(x)
		values = self.W_Value(x)

		print("Before view and head split shape:",queries.shape)
		print("Before view and head split:",queries)

		queries = queries.view(batch_size, context_length, self.num_heads, self.head_dim)
		print("After view and head split shape:", queries.shape)
		print("After view and head split:",queries)
		keys = keys.view(batch_size, context_length, self.num_heads, self.head_dim)
		values = values.view(batch_size, context_length, self.num_heads, self.head_dim)

		queries = queries.transpose(1,2)
		print(queries.shape)
		keys = keys.transpose(1,2)
		values = values.transpose(1,2)

		attention_scores = queries @ keys.transpose(2,3)

		attention_scores.masked_fill_(self.mask.bool()[:context_length, :context_length], -torch.inf)

		attention_weights = torch.softmax(attention_scores, dim=-1)
		attention_weights = self.dropout(attention_weights)

		context_vectors = (attention_weights @ values).transpose(1,2)
		context_vectors = context_vectors.contiguous().view(batch_size, context_length, self.d_out)

		context_vectors = self.out_project(context_vectors)

		return context_vectors

def main():
	torch.manual_seed(123)
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
		"emb_dim": 4,
		"context_length": 4,
		"dropout": 0.5
	}

	emb_layer = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
	positional_layer = torch.nn.Embedding(config["context_length"], config["emb_dim"])

	token_embeddings = emb_layer(torch.tensor(token_ids))
	positional_embeddings = positional_layer(torch.arange(config["context_length"]))

	input_embeddings = token_embeddings + positional_embeddings

	multi_head_v1 = MultiHeadAttention(
		din=config["emb_dim"], 
		dout=config["emb_dim"],
		dropout=config["dropout"], 
		context_length=config["context_length"], 
		qkv_bias=True,
		num_heads=2
	)
	context_vectors = multi_head_v1(input_embeddings.unsqueeze(0))
	print(context_vectors)
	print("Shape:", context_vectors.shape)

main()



