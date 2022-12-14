import numpy as np
import torch
from torch.nn.functional import softmax


def sample(predictions):
    return max(predictions).index[predictions]


# Start with output of a decoder block - matrix, each row a word depth of d
l = 14
d = 50
r = np.ones((l,d))

# Also need the vocab embedding which is shape (k, d)
k = 10000
vocab_embedding = np.ones((k,d))

# Matmul together to get logits across each word (column) for each input (row)
logits = np.matmul(r, vocab_embedding.T)

# Softmax and sample original vocab
predictions = softmax(logits)
h_index = sample(predictions)
h_word = words[h_index]

