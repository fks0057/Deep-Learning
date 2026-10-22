import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=4,
    embedding_dim=3
)

tokens = torch.tensor([0, 1, 2, 3])

x = embedding(tokens)

print(x)
print(x.shape)