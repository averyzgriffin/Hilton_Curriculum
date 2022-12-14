import numpy as np
import torch

# Start with the projected output from the self-attention block
l = 14
d = 50
z = np.ones((l, d))  # each row is the filter values of a word

# Model
layer1 = np.ones((z.shape[1], z.shape[1]*4))  # first layer is 4x the size of z
layer2 = np.ones((z.shape[1], d))  # Just projects back to the original size d

a = np.matmul(z, layer1)
r = np.matmul(a, layer2)



