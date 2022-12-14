import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax


# Reminder f and d are probably going to be equivalent for me (ignoring heads for now)
class GPTAve(nn.Module):
    def __init__(self, num_decoders, d, f):
        super(GPTAve, self).__init__()

        self.num_decoders = num_decoders
        self.d = d  # The original depth of the embedding for each token.
        self.f = f  # The depth of each token after being projected during attention. Usually just d or smaller.
        self.heads = 1
        self.f /= self.heads  # split the projection into each head
        self.model = self.build_model()

    def forward(self, x):
        for decoder in self.model:
            x = decoder(x)

    def build_model(self):
        model = []
        for n in range(self.num_decoders):
            model.append(Decoder(self.d, self.f))
        return model


class Decoder(nn.Module):
    def __init__(self, d, f):
        super(Decoder, self).__init__()

        self.attention_block = Attention(d, f)
        self.feedforward_block = FeedForward(d, f)
        self.residuals = []  # TODO the residuals are just the value before the previous block

    def forward(self, x):
        x = self.attention_block(x)
        x = self.layer_normalization(x + self.residuals)
        x = self.feedforward_block(x)
        x = self.layer_normalization(x + self.residuals)
        return x

    # TODO Define this
    @staticmethod
    def layer_normalization(x):
        return x


class Attention(nn.Module):
    def __init__(self,d, f):
        super(Attention, self).__init__()

        self.f = f  # needed for the normalization of the filter. todo I think I can remove this. see compute filter
        self.wq = nn.Linear(d, f)  # size is (d,f) bc we project d into f
        self.wk = nn.Linear(d, f)
        self.wv = nn.Linear(d, f)
        self.attention_filter = torch.Tensor()  # I don't think I need to define this ahead of time or as parameters

    def forward(self, x):
        q,k,v = self.comptue_qkv(x)
        self.compute_attention_filter(q, k)
        self.mask_attention_filter()
        self.softmax_attention_filter()
        x = self.apply_attention_filter(v)
        x = self.project_filtered_values(x)
        return x

    def comptue_qkv(self, x):
        q = self.wq(x)  # (l,f); l for each word, f projection
        k = self.wk(x)
        v = self.wv(x)
        return q,k,v

    def compute_attention_filter(self, q, k):
        self.attention_filter = torch.matmul(q, k.T) / np.sqrt(self.f)  # todo I should be able to replace f with q.shape[1]

    def mask_attention_filter(self):
        mask = torch.zeros_like(self.attention_filter)
        indices = torch.triu_indices(self.attention_filter.shape[0], 1)
        mask[indices] = torch.Tensor(-np.inf)
        self.attention_filter = self.attention_filter + mask

    def softmax_attention_filter(self):
        self.attention_filter = softmax(self.attention_filter)

    def apply_attention_filter(self, v):
        return torch.matmul(self.attention_filter, v)  # lxf

    def projection_layer(self, x):
        proj_z = nn.Linear(self.f, self.f)  # fxf todo maybe it's supposed to be fxd. not sure
        x = proj_z(x)  # lxf
        return x


class FeedForward(nn.Module):
    def __init__(self, d, f):
        super(FeedForward, self).__init__()

        self.layer1 = nn.Linear(f, f*4)  # first layer is 4x the size of the output of the attention block so f*4
        self.layer2 = nn.Linear(f*4, d)  # Just projects back to the original depth d

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x







