import pandas as pd
import numpy as np
import torch
from torchtext.data import get_tokenizer
from torchtext.transforms import BERTTokenizer
from nltk.corpus import reuters
from transformers import AutoTokenizer, TFAutoModel, AutoModel, pipeline

from pos_encode import getPositionEncoding


def get_embedding_matrix(vocab_dict, embedding_dict, dims):
    embedding_matrix = np.zeros((len(vocab_dict) + 1, dims))
    missing_words = []
    for word, index in vocab_dict.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
        else:
            missing_words.append(word)

    print("missing words ", len(missing_words))
    return embedding_matrix


# Large Corpus
# with open("corpuses/raw_text.txt") as f:
#     corpus = f.read().lower()
# tokenizer = get_tokenizer("basic_english")
# tokens = tokenizer(corpus)

# corpus = "was an electrical engineer and inventor. He is most known as the inventor of the transform."
# corpus = corpus.lower()

corpus = "My my my name name is Avery"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel .from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
data = pipe(corpus)

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # TODO Figure out how to get rid of or locate the ##s
# tokens = tokenizer.tokenize(corpus)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# glove = pd.read_csv('embeddings/glove.6B.50d.txt', sep=" ", quoting=3, header=None, index_col=0)
# glove_embedding = {key: val.values for key, val in glove.T.items()}
# vocab_index = dict([(y,x+1) for x,y in enumerate(sorted(set(tokens)))])

# embedding_matrix = get_embedding_matrix(vocab_index, glove_embedding, 50)
# pos_matrix = getPositionEncoding(embedding_matrix.shape[0], embedding_matrix.shape[1])
# embedding_matrix += pos_matrix

# embedding_tensor = torch.nn.Embedding(num_embeddings=len(vocab_index), embedding_dim=50)
# embedding_tensor.weight = torch.nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
# embedding_tensor.weight.requires_grad = False



# corpus = load_corpus()  # list of words, say 1M words taken from a bunch of sources like books. contiguous
# tokens = tokenize(corpus)  # list of n tokens; each word is split into unique tokens using tokenization
# token_ints = convert_tokens(tokens)  # list of n ints; each token is converted to a unique integer i.d.
# embedding = embed(token_ints)  # n x e matrix; n = # of tokens, e = # of dimensions in embedding space
# minibatch = batch(embedding)  # b x e matrix; b = size of batch. group the embedding vectors into batches of e.g. 512 contiguous tokens 512 x 768
# pos_encode = add_pos(minibatch)  # b x e matrix; nothing changes, just added a vector of same size (512 x 768)

print()