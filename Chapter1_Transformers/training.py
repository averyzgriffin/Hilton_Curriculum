import torch
from torchtext.datasets import WikiText2
from transformers import AutoTokenizer, AutoModel, pipeline
from pos_encode import getPositionEncoding

from models import GPTAve


def preprocess_text(text, token_pipe: pipeline, pos_matrix):
    x = token_pipe(text)[0]
    # pos_matrix = getPositionEncoding(max_len, len(x[0]))
    x = x + pos_matrix[:len(x)]
    return torch.Tensor(x[:-1])  # TODO I am concerned about ignoring the last input


def get_label(text, tokenizer, max_sequence):
    token_id = torch.tensor(tokenizer.encode(text, max_length=max_sequence, truncation=True)[1:])
    return token_id
    # return torch.nn.functional.one_hot(token_id, tokenizer.vocab_size).float()


device = torch.device("cuda:0")

train = WikiText2(split='train')
max_sequence = 256
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=max_sequence)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
embedding_model = AutoModel.from_pretrained("gpt2")
embedding_model.resize_token_embeddings(len(tokenizer))
pipe = pipeline('feature-extraction', model=embedding_model, tokenizer=tokenizer, padding=True, truncation=True)

pos_matrix = getPositionEncoding(max_sequence, embedding_model.embed_dim)  # we are just computing this once now

model = GPTAve(num_decoders=3, d=embedding_model.embed_dim, f=embedding_model.embed_dim,
               heads=12, embedding_matrix=embedding_matrix).to(device)
print(model)
loss_fc = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(lr=1e-5, params=model.parameters())

s = 0
for text in train:
    x = preprocess_text(text, pipe, pos_matrix).to(device)
    if len(x) > 1:
        y = get_label(text, tokenizer, max_sequence).to(device)
        opt.zero_grad()  # clears gradients
        logits = model(x)
        loss = loss_fc(logits, y)
        loss.backward()  # compute gradients
        opt.step()  # update weights
        print(f"Step {s} Loss {loss}")
        s += 1



