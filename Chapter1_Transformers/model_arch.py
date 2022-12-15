import numpy as np


"""class Transformer:

	def __init__():


	def pipeline():

		# Converting input to embedding vectors
		corpus = load_corpus()  # list of words, say 1M words taken from a bunch of sources like books. contiguous
		tokens = tokenize(corpus)  # list of n tokens; each word is split into unique tokens using tokenization
		token_ints = convert_tokens(tokens)  # list of n ints; each token is converted to a unique integer i.d.
		embedding = embed(token_ints)  # n x e matrix; n = # of tokens, e = # of dimensions in embedding space
		minibatch = batch(embedding)  # b x e matrix; b = size of batch. group the embedding vectors into batches of e.g. 512 contiguous tokens 512 x 768
		pos_encode = add_pos(minibatch)  # b x e matrix; nothing changes, just added a vector of same size (512 x 768)
	

		# DECODER
		# Attention
		q = linear(embedding.copy())  # b x f matrix; f = dimension we squeeze e into if any (usually not?); split by h if using heads
		k = linear(embedding.copy())  # b x f matrix; f = dimension we squeeze e into if any (usually not?); split by h if using heads
		v = linear(embedding.copy())  # b x f matrix; f = dimension we squeeze e into if any (usually not?); split by h if using heads
		attention_filter = np.dot(q.T,k) / # TODO  # b x b matrix
		masked_filter = mask(attention_filter)   # b x b matrix; everything above the diagonal is zeroed out
		filtered_embedding = mat(masked_filter, v)  # b x f matrix; concatenated to keep f if using heads
		projected_decoding = linear(filtered_embedding)  # b x f matrix; not sure on purpose even after reading a lot, maybe to train weights to convert awkward concatenated attention-heads output to usable input for MLP mechanism
		# Residuals and Normalize
		projected_decoding = normalize(residuals + projected_decoding)
		# Feed Forward
		mlp_output = MLP(projected_decoding)  # 2 layers, 1st = 4x size of f for arbitrary reasons, 2nd = project back into size f; purpose still unclear outside of extract information from attention block
		# Residuals and Normalize
		decoder_output = normalize(residuals + mlp_output)


		# Final Prediction
		predictions = softmax(linear(decoder_output))  # k x 1 vector, k = # of words in the model vocabulary
"""









