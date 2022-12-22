import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
# import pandas as pd
# import altair as alt
# from torchtext.data.functional import to_map_style_dataset
# from torch.utils.data import DataLoader
# from torchtext.vocab import build_vocab_from_iterator
# import torchtext.datasets as datasets
# import spacy
# #import GPUtil
# import warnings
# from torch.utils.data.distributed import DistributedSampler
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

from model_arch import make_model
from training_setup import LabelSmoothing, rate, run_epoch, data_gen,\
    SimpleLossCompute, greedy_decode, DummyOptimizer, DummyScheduler


# Simple copy task: given an input of token, spit out the same tokens
def example_simple_model():
    V = 11  # I believe V is "size of the vocab". GPT2 had V=52k for example.
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))



if __name__ == "__main__":
    example_simple_model()













