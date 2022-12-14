import numpy as np
import torch
from torchtext.datasets import WikiText2
from transformers import AutoTokenizer, AutoModel, pipeline


train = WikiText2(split='train')

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel .from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

def create_embedding_tensor(embedding):
    embedding_tensor = torch.nn.Embedding(num_embeddings=len(embedding[0]), embedding_dim=len(embedding[0][0]))
    embedding_tensor.weight = torch.nn.Parameter(torch.tensor(embedding[0],dtype=torch.float32))
    embedding_tensor.weight.requires_grad = False
    return embedding_tensor

def compute_loss():
    return

def update_weights():
    return

for text in train:
    text = pipe(text)
    x = create_embedding_tensor(text)
    prediction = model(x)
    loss = compute_loss
    update_weights()

