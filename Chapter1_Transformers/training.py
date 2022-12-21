import torch
from torchtext.datasets import WikiText2
from transformers import AutoTokenizer, AutoModel, pipeline, GPT2TokenizerFast
from pos_encode import getPositionEncoding

from models import GPTAve
from helpers import NoamOpt, get_std_opt


def preprocess_text(text, token_pipe: pipeline, pos_matrix):
    x = token_pipe(text)[0]
    # pos_matrix = getPositionEncoding(max_len, len(x[0]))
    x = x + pos_matrix[:len(x)]
    return torch.Tensor(x[:-1])  # TODO I am concerned about ignoring the last input

def preprocess_text2(ids, model, pos_matrix):
    x = model(torch.tensor(ids))[0]
    x = x.detach().numpy() + pos_matrix[:len(x)]
    return torch.Tensor(x[:-1])  # TODO I am concerned about ignoring the last input


def get_label(text, tokenizer, max_sequence):
    token_id = torch.tensor(tokenizer.encode(text, max_length=max_sequence, truncation=True)[1:])
    return token_id
    # return torch.nn.functional.one_hot(token_id, tokenizer.vocab_size).float()


with open("corpuses/raw_text.txt", "r") as f:
    text = f.read()


device = torch.device("cuda:0")

# train = WikiText2(split='train')

max_sequence = 512
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=max_sequence)
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=False)#, model_max_length=max_sequence)
embedding_model = AutoModel.from_pretrained("gpt2")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # TODO pretty sure this isn't doing anything useful
# embedding_model.resize_token_embeddings(len(tokenizer))  # TODO pretty sure this isn't doing anything useful
embedding_matrix = torch.clone(embedding_model.wte.weight.detach()).to(device)
pipe = pipeline('feature-extraction', model=embedding_model, tokenizer=tokenizer,
                padding=True, truncation=True)

pos_matrix = getPositionEncoding(max_sequence, embedding_model.embed_dim)  # we are just computing this once now

model = GPTAve(num_decoders=4, d=embedding_model.embed_dim, f=embedding_model.embed_dim,
               heads=8, embedding_matrix=embedding_matrix).to(device)

for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

print(model)
loss_fc = torch.nn.CrossEntropyLoss()
# opt = torch.optim.Adam(lr=.000001, params=model.parameters())
opt = get_std_opt(model)  # Taken from Annotated Transformer

# Use this if not using pipeline / want uniform sample sizes
tokens = tokenizer(text)
ids = tokens["input_ids"]
sequences = [ids[i:i+max_sequence] for i in range(0, len(ids), max_sequence)]
random.shuffle(sequences)

s = 0
# for text in train:
for seq in sequences:
    # Old Block
    # x = preprocess_text2(seq, embedding_model, pos_matrix).to(device)
    # # x = preprocess_text(text, pipe, pos_matrix).to(device)
    # # if len(x) > 1:
    # y = torch.tensor(seq[1:]).to(device)
    # # y = get_label(text, tokenizer, max_sequence).to(device)
    # opt.zero_grad()  # clears gradients
    # logits = model(x)
    # loss = loss_fc(logits, y)
    # loss.backward()  # compute gradients
    # opt.step()  # update weights
    # print(f"Step {s} Loss {loss}")
    # s += 1

    # x = preprocess_text(text, pipe, pos_matrix).to(device)
    x = preprocess_text2(seq, embedding_model, pos_matrix).to(device)  # For when we are not using pipeline
    # if (s+1) % 16 == 0:
    #     opt.step()  # update weights
    #     opt.optimizer.zero_grad()  # clears gradients
    if len(x) > 1:
        # y = get_label(text, tokenizer, max_sequence).to(device)  # Pipeline
        y = torch.tensor(seq[1:]).to(device)  # Not using pipeline
        # y = torch.ones((len(x)), dtype=int).to(device) * 13  # Label is just all ones
        # y = torch.tensor(seq[:-1]).to(device)  # Label is just the sequence shifted up one
        decoded_labels = [tokenizer.decode(y[i]) for i in range(len(y))]
        opt.optimizer.zero_grad()  # clears gradients
        logits = model(x)
        predictions = [torch.argmax(logits[i]).detach().tolist() for i in range(len(logits))]
        decoded_predictions = [tokenizer.decode(predictions[i]) for i in range(len(predictions))]
        loss = loss_fc(logits, y)
        loss.backward()  # compute gradients
        opt.step()  # update weights

        print(f"\nStep {s} Loss {loss} LR {opt._rate}")
        # print("Tokens    ", tokenizer.tokenize(text[:30]))
        print("Tokens    ", tokenizer.decode(seq[:30]))
        print(f"Labels Ids     {y.detach().tolist()[:15]}")
        print(f"Label Tokens       {decoded_labels[:15]}")
        print(f"Prediction Ids {predictions[:15]}")
        print(f"Prediction Tokens  {decoded_predictions[:15]}")
        s += 1



