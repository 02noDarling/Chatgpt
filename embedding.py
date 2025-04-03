import torch
import torch.nn as nn
import numpy as np
import math
from config import *

class EmbeddingWithPosition(nn.Module):
    def __init__(self):
        super(EmbeddingWithPosition, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, DIM)
        self.position_embedding = torch.zeros(MAX_SEQ_LEN, DIM)

        for i in range(MAX_SEQ_LEN):
            for j in range(DIM):
                if j%2==0:
                    temp = torch.tensor(-(j/DIM * math.log(10000)))
                    self.position_embedding[i][j] = torch.sin(i * torch.exp(temp))
                else:
                    temp = torch.tensor(-((j-1)/DIM * math.log(10000)))
                    self.position_embedding[i][j] = torch.cos(i * torch.exp(temp))
    
    def forward(self, x): # x:[batch, seq_len]
        x = self.embedding(x)
        return x + self.position_embedding.unsqueeze(0).expand(x.shape[0],-1,-1)[:,:x.shape[1],:]

if __name__ == "__main__":
    embedding = EmbeddingWithPosition()

    input = torch.randint(0,11,(2,10))
    print(input)
    output = embedding(input)
    print(output.shape)