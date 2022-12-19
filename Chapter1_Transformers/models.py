import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax


# Reminder f and d are probably going to be equivalent for me (ignoring heads for now)
class GPTAve(nn.Module):
    def __init__(self, num_decoders, d, f, heads, embedding_matrix):
        super(GPTAve, self).__init__()

        self.num_decoders = num_decoders
        self.d = d  # The original depth of the embedding for each token.
        self.f = f  # The depth of each token after being projected during attention. Usually just d or smaller.
        self.heads = heads
        self.embedding_matrix = embedding_matrix
        self.model = self.build_model()

    def forward(self, x):
        for decoder in self.model:
            x = decoder(x)
        x = self.compute_logits(x)
        return x

    def build_model(self):
        model = torch.nn.ModuleList()
        for n in range(self.num_decoders):
            model.append(Decoder(self.d, self.f, self.heads))
        return model

    def compute_logits(self, x):
        x = torch.matmul(x, self.embedding_matrix.T)
        return softmax(x, dim=1)


class Decoder(nn.Module):
    def __init__(self, d, f, heads):
        super(Decoder, self).__init__()

        self.attention_block = Attention(d, f, heads)  # result is lxf
        self.feedforward_block = FeedForward(d, f)  # result is lxd I think
        self.LayerNorm1 = nn.LayerNorm(f)  # TODO I might need l here
        self.LayerNorm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor):
        residuals = x.detach().clone()
        x = self.attention_block(x)
        x = self.LayerNorm1(x + residuals)
        residuals = x.detach().clone()
        x = self.feedforward_block(x)
        x = self.LayerNorm2(x + residuals)
        return x


class Attention(nn.Module):
    def __init__(self, d, f, heads):
        super(Attention, self).__init__()

        self.wq = nn.Linear(d, f)  # size is (d,f) bc we project d into f
        self.wk = nn.Linear(d, f)
        self.wv = nn.Linear(d, f)
        self.heads = heads
        self.attention_filter = torch.Tensor()  # TODO I think I may need to explicitely say gradient=False to avoid treating these as parameters / not define it ahead of time
        self.proj_z = nn.Linear(f, f)  # fxf (maybe it's supposed to be fxd. not sure)

    def forward(self, x):
        q,k,v = self.compute_qkv(x)
        q,k,v = self.split_heads(q, k, v)
        self.compute_attention_filter(q, k)
        self.mask_attention_filter()
        self.softmax_attention_filter()
        x = self.apply_attention_filter(v)
        x = self.concatenate_heads(x)
        x = self.projection_layer(x)
        return x

    def compute_qkv(self, x):
        q = self.wq(x)  # (l,f); l for each word, f projection
        k = self.wk(x)
        v = self.wv(x)
        return q,k,v

    def split_heads(self, q, k, v):
        q = torch.stack(torch.split(q, (q.shape[1]//self.heads), dim=1), dim=0)
        k = torch.stack(torch.split(k, (k.shape[1]//self.heads), dim=1), dim=0)
        v = torch.stack(torch.split(v, (v.shape[1]//self.heads), dim=1), dim=0)
        return q,k,v

    def compute_attention_filter(self, q, k):
        self.attention_filter = torch.bmm(q, torch.permute(k, (0, 2, 1)))

    def mask_attention_filter(self):
        mask = torch.zeros_like(self.attention_filter)
        indices = torch.triu_indices(self.attention_filter.shape[0], self.attention_filter.shape[1], offset=1)
        mask[:, indices[0], indices[1]] = -1000000000
        self.attention_filter = self.attention_filter + mask

    def softmax_attention_filter(self):
        self.attention_filter = softmax(self.attention_filter, dim=2)

    def apply_attention_filter(self, v):
        return torch.bmm(self.attention_filter, v)  # hxlxf

    @staticmethod
    def concatenate_heads(x):
        return x.transpose(0,1).reshape(x.shape[1], x.shape[0]*x.shape[2])  # reshapes hxlx(f/h) into lxf)

    def projection_layer(self, x):
        x = self.proj_z(x)  # lxf
        return x


class FeedForward(nn.Module):
    def __init__(self, d, f):
        super(FeedForward, self).__init__()

        self.layer1 = nn.Linear(f, f*4)  # first layer is 4x the size of the output of the attention block so f*4
        self.layer2 = nn.Linear(f*4, d)  # Just projects back to the original depth d
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)  # TODO I was missing this
        x = self.layer2(x)
        return x







