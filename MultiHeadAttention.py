import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, nhead, dropout=0.1, batch_first=True):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.k_dim = model_dim // nhead
        self.nhead = nhead
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)
        self.dropout =nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = query.shape
        Q = self.q_linear(query).reshape(batch_size, seq_len, self.nhead, self.k_dim).transpose(1, 2)
        K = self.k_linear(key).reshape(batch_size, seq_len, self.nhead, self.k_dim).transpose(1, 2)
        V = self.v_linear(value).reshape(batch_size, seq_len, self.nhead, self.k_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(self.k_dim)
        if attn_mask != None:
            scores += attn_mask
        if key_padding_mask != None:
            scores.masked_fill(key_padding_mask.bool().unsqueeze(1).unsqueeze(1), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V).transpose(1, 2).reshape(batch_size, seq_len, self.model_dim)
        output = self.out_linear(output)
        return output, attn

if __name__ == "__main__":
    batch_size = 10
    seq_len = 150
    self_attn = MultiHeadAttention(DIM, GPT_HEAD, dropout=0.1, batch_first=True)
    X = torch.randn(batch_size, seq_len, DIM)
    attn_mask = torch.full((seq_len, seq_len), 0)
    key_padding_mask = torch.randint(0, 2, (batch_size, seq_len))
    output, attn_output_weights = self_attn(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    print(output.shape,attn_mask.shape)

