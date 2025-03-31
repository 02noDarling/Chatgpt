import torch
import torch.nn as nn
import numpy as np
import math

class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, dim, max_seq_len):
        super(EmbeddingWithPosition, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = torch.zeros(max_seq_len, dim)

        for i in range(max_seq_len):
            for j in range(dim):
                if j%2==0:
                    temp = torch.tensor(-(j/dim * math.log(10000)))
                    self.position_embedding[i][j] = torch.sin(i * torch.exp(temp))
                else:
                    temp = torch.tensor(-((j-1)/dim * math.log(10000)))
                    self.position_embedding[i][j] = torch.cos(i * torch.exp(temp))
    
    def forward(self, x): # x:[batch, seq_len]
        x = self.embedding(x)
        return x + self.position_embedding.unsqueeze(0).expand(x.shape[0],-1,-1)[:,:x.shape[1],:]

if __name__ == "__main__":
    embedding = EmbeddingWithPosition(11, 5, 11)

    input = torch.randint(0,11,(2,10))
    print(input)
    output = embedding(input)
    print(output)