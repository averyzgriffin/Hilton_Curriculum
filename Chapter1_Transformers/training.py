import numpy as np
import torch
from torchtext.datasets import WikiText2
from transformers import AutoTokenizer, AutoModel, pipeline
from pos_encode import getPositionEncoding

from models import GPTAve, Decoder, Attention, FeedForward


def preprocess_text(text, token_pipe):
    x = token_pipe(text)[0]
    pos_matrix = getPositionEncoding(len(x), len(x[0]))  # Todo concerned about samples of different sizes
    # x = create_embedding_tensor(text)
    # y = torch.Tensor(x[1:])
    return torch.Tensor(x[:-1] + pos_matrix[:-1])


def get_label(text, tokenizer):
    token_id = torch.tensor(tokenizer.encode(text)[1:])
    return torch.nn.functional.one_hot(token_id, tokenizer.vocab_size).float()


def get_top_predictions(logits, k):
    # sort(logits)
    # extract_tokens(logits)
    return logits


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
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
embedding_model = AutoModel.from_pretrained("gpt2")
embedding_model.resize_token_embeddings(len(tokenizer))
pipe = pipeline('feature-extraction', model=embedding_model, tokenizer=tokenizer)

model = GPTAve(num_decoders=1, d=embedding_model.embed_dim, f=embedding_model.embed_dim, vocab_size=tokenizer.vocab_size)
loss_fc = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(lr=1e-3, params=model.parameters())

s = 0
for text in train:
    x = preprocess_text(text, pipe)
    y = get_label(text, tokenizer)
    opt.zero_grad()  # clears gradients
    logits = model(x)
    # predictions = get_top_predictions(logits, k=40)
    loss = loss_fc(logits, y)
    loss.backward()  # compute gradients
    opt.step()  # update weights
    print(f"Step {s} Loss {loss}")
    s += 1



