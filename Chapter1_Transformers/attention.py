import numpy as np
import torch
from torch.nn.functional import softmax


# Start with the encoding of some batch. Let's use our example from before
l = 14
d = 50  # depth of the embedding vectors
embedding = np.ones((l, d))  # 14 words in the sequence, each word has depth of d

# Create 3 matrices to be used for calculating query, key, value. These are trainable.
# Should we just make this a 14,d,d tensor? NO! EACH WORD JUST PRODUCES A NEW ROW IN THE OUTPUT VIA MATMUL. AUTOMATED.
h = 1  # number of heads
f = d  # what we project the embedding vector to. the length of each query, value, key. usually smaller than d
f = f // h  # we split into each head
wq = np.ones((d, f))  # size is (d,f) because we need d for the matmul to work and then f is the projection
wk = np.ones((d, f))
wv = np.ones((d, f))

#  Multiply weight matrices by the original embedding. Same as passing through a linear layer or dot product
#  Each result is l x f matrix; l - a row for each word. f - whatever the projection is, just d for now.
q = np.matmul(embedding, wq)
k = np.matmul(embedding, wk)
v = np.matmul(embedding, wv)

# Calculate the attention filter
# This operation is non-intuitive conceptually but it's math is simple. Each query is multiplied with each key
# Each pairing produces a single score. Since there are l queries and l keys, the attention filter is lxl
# In practice, it's a little tricky since each query/key is split into f dimensions. This is handled by
# simple matrix multiplying where one of q or k is transposed (depending on how q and k are constructed)
attention_filter = np.matmul(q, k.T)

# Normalize the filter
attention_filter /= np.sqrt(f)  # TODO I think f is correct but not sure

# Mask out the attention filter
mask = np.zeros_like(attention_filter)
indices = np.triu_indices(attention_filter.shape[0], 1)
mask[indices] = -np.inf
attention_filter = attention_filter + mask

# Softmax the Attention Filter - softmax each query (row)
attention_filter = torch.Tensor.numpy(softmax(torch.from_numpy(attention_filter)))

# Apply the filter to the original embedding matrix (the values)
# The most complicated math of the whole thing. I really don't know how to explain this one in a easy to remember way
# Basically each row of scores (each query or each word) is dotted into the entire value matrix.
# This produces a single row vector z; this is the filtered values for the first word (first row). z has length f
# Basically each column that S (attention filter) is dotted into of V produces another column in z. V has f columns
# Then, we just repeat for each row in S to produce additional rows in Z. We can do this all at once by
# matmul s and V (no transposing, just make sure S has queries as rows and V has words as rows)
z = np.matmul(attention_filter, v)  # lxf

# Project the filtered values to a vector of the same shape
# I think the purpose here is to generally allow the model to learn how to interpret the attention output
# and specifically handle multiple heads since this is where you would concatenate different heads
proj_z = np.ones((f,f))  # fxd
z = np.matmul(z, proj_z)  # lxd - projects z back to the original embedding size d

print('hello')


