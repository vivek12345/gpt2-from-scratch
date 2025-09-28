import tiktoken
import torch

def text_to_token_ids(text, tokenizer):
	# get token ids => as a list [464, 6193, 318, 2089]
	encoded = tokenizer.encode(text)

	# let's make it a tensor for llm operations => tensor([ 464, 6193,  318, 2089])
	token_ids = torch.tensor(encoded)

	# finally let's make it like a batch with only 1 batch > tensor([[ 464, 6193,  318, 2089]])
	return token_ids.unsqueeze(0)

# we get token ids as a tensor([[ 464, 6193,  318, 2089]]) with batch
def token_ids_to_text(token_ids, tokenizer):
	# first remove the batch and process normally => tensor([ 464, 6193,  318, 2089])
	token_ids = token_ids.squeeze(0)

	# make it back to a list to tiktoken can process => [ 464, 6193,  318, 2089]
	encoded = token_ids.tolist()

	# finally decode it to back to wokrs => The weather is bad
	decoded = tokenizer.decode(encoded)

	return decoded


tokenizer = tiktoken.get_encoding("gpt2")
text = "The weather is bad"

token_ids = text_to_token_ids(text, tokenizer)
print(token_ids)

decoded = token_ids_to_text(token_ids, tokenizer)
print(decoded)
