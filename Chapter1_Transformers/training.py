import torch
from torchtext.datasets import WikiText2
from transformers import AutoTokenizer, AutoModel, pipeline
from pos_encode import getPositionEncoding

from models import GPTAve


def preprocess_text(text, token_pipe):
    x = token_pipe(text)[0]
    pos_matrix = getPositionEncoding(len(x), len(x[0]))  # Todo concerned about samples of different sizes
    # x = create_embedding_tensor(text)  # I don't think we need this anymore but just in case
    return torch.Tensor(x[:-1] + pos_matrix[:-1])


def get_label(text, tokenizer):
    token_id = torch.tensor(tokenizer.encode(text)[1:])
    return torch.nn.functional.one_hot(token_id, tokenizer.vocab_size).float()


train = WikiText2(split='train')

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
embedding_model = AutoModel.from_pretrained("gpt2")
embedding_model.resize_token_embeddings(len(tokenizer))
pipe = pipeline('feature-extraction', model=embedding_model, tokenizer=tokenizer)

device = torch.device("cuda:0")
model = GPTAve(num_decoders=1, d=embedding_model.embed_dim, f=embedding_model.embed_dim, vocab_size=tokenizer.vocab_size).to(device)
print(model)
loss_fc = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(lr=1e-3, params=model.parameters())

s = 0
for text in train:
    x = preprocess_text(text, pipe).to(device)
    y = get_label(text, tokenizer).to(device)
    opt.zero_grad()  # clears gradients
    logits = model(x)
    loss = loss_fc(logits, y)
    loss.backward()  # compute gradients
    opt.step()  # update weights
    print(f"Step {s} Loss {loss}")
    s += 1



