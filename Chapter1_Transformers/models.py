import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax


# Reminder f and d are probably going to be equivalent for me (ignoring heads for now)
class GPTAve(nn.Module):
    def __init__(self, num_decoders):
        super(GPTAve, self).__init__()

        self.num_decoders = num_decoders


class Decoder:
    def __init__(self):
        pass

    def projection_layer(self):
        proj_z = np.ones((self.f, self.f))  # fxd
        self.z = np.matmul(self.z, proj_z)  # lxd - projects z back to the original embedding size d


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.heads = 1
        self.f = 756  # what we project x to. length of q,v,k. usually d or smaller
        self.f /= self.heads  # split the projection into each head
        self.wq = []  # size is (d,f) bc we project d into f
        self.wk = []
        self.wv = []
        self.attention_filter = []

    def forward(self, x):
        q,k,v = self.comptue_qkv(x)
        self.compute_attention_filter(q, k)
        self.mask_attention_filter()
        self.softmax_attention_filter()
        x = self.apply_attention_filter(v)
        return x

    def comptue_qkv(self, x):
        q = np.matmul(x, self.wq)  # (l,f); l for each word, f projection
        k = np.matmul(x, self.wk)
        v = np.matmul(x, self.wv)
        return q,k,v

    def compute_attention_filter(self, q, k):
        self.attention_filter = np.matmul(q, k.T) / np.sqrt(self.f)

    def mask_attention_filter(self):
        mask = np.zeros_like(self.attention_filter)
        indices = np.triu_indices(self.attention_filter.shape[0], 1)
        mask[indices] = -np.inf
        self.attention_filter = self.attention_filter + mask

    def softmax_attention_filter(self):
        self.attention_filter = torch.Tensor.numpy(softmax(torch.from_numpy(self.attention_filter)))

    def apply_attention_filter(self, v):
        return np.matmul(self.attention_filter, v)  # lxf


class FeedForward(nn.Module):
    def __init__(self, d, f):
        super(FeedForward, self).__init__()

        self.d = d  # The original depth of the embedding for each token.
        self.f = f  # The depth of each token after being projected during attention. Usually just d or smaller.
        self.layer1 = nn.Linear(f, f*4)  # first layer is 4x the size of the output of the attention block so f*4
        self.layer2 = nn.Linear(f*4, d)  # Just projects back to the original depth d

    def forward(self, x):
        x = np.matmul(x, self.layer1)
        x = np.matmul(x, self.layer2)
        return x







