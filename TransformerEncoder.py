import torch
import torch.nn as nn
from config import *
from MultiHeadAttention import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, nhead, feedforward_dim, dropout=0.1, batch_first=True):
        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout, batch_first=batch_first)
        self.self_attn = MultiHeadAttention(model_dim, nhead, dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.activation = nn.ReLU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 多头自注意力层
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # 前馈网络
        feedforward_output = self.linear1(src)
        feedforward_output = self.activation(feedforward_output)
        feedforward_output = self.dropout(feedforward_output)
        feedforward_output = self.linear2(feedforward_output)

        src = src + self.dropout2(feedforward_output)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, nums):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for i in range(nums)])
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src
    
if __name__ == "__main__":
    encoder_layer = TransformerEncoderLayer(DIM, GPT_HEAD, GPT_FF, dropout=0.1, batch_first=True)
    encoder = TransformerEncoder(encoder_layer, GPT_BLOCKS)
    src = torch.randn(10, 5, DIM)
    output = encoder(src)
    print(output.shape)