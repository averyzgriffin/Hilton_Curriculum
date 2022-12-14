import numpy as np
import torch
from torchtext.datasets import WikiText2
from transformers import AutoTokenizer, AutoModel, pipeline
from pos_encode import getPositionEncoding

from models import GPTAve, Decoder, Attention, FeedForward


def create_embedding_tensor(embedding):
    embedding_tensor = torch.nn.Embedding(num_embeddings=len(embedding), embedding_dim=len(embedding[0]))
    embedding_tensor.weight = torch.nn.Parameter(torch.tensor(embedding,dtype=torch.float32))
    embedding_tensor.weight.requires_grad = False
    return embedding_tensor


def project_to_embedding(z):
    return z


def compute_loss(x):
    return x


train = WikiText2(split='train')

tokenizer = AutoTokenizer.from_pretrained("gpt2")
embedding_model = AutoModel.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
embedding_model.resize_token_embeddings(len(tokenizer))
pipe = pipeline('feature-extraction', model=embedding_model, tokenizer=tokenizer)

model = GPTAve(num_decoders=1)

for text in train:
    text = pipe(text)[0]
    # TODO add position encoding here
    x = create_embedding_tensor(text)
    z = model(x)
    prediction = project_to_embedding(z)
    loss = compute_loss(prediction)
    loss.backward()  # update weights

