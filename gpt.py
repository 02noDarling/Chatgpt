import torch
import torch.nn as nn
from embedding import EmbeddingWithPosition
from config import *
import torch.nn.functional as F
import random
from tokenizer import BPETokenizer

class GPT(nn.Module):
    def __init__(self, vocab_size, dim, max_seq_len, nhead, feedforward, blocks):
        super(GPT, self).__init__()
        self.embedding = EmbeddingWithPosition(vocab_size, dim, max_seq_len)

        # 用encoder加掩码的方式实现decoder-only
        decoder_layer = nn.TransformerEncoderLayer(dim, nhead, feedforward, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, blocks)
        self.mask = nn.Transformer.generate_square_subsequent_mask(max_seq_len)
        self.prob_linear = nn.Linear(dim, vocab_size)
    
    def forward(self, input_ids, padding_mask=None): # x: [batch, seq_len]
        input_ids = self.embedding(input_ids)
        seq_len = input_ids.shape[1]
        mask =self.mask[:seq_len,:seq_len]
        output = self.decoder(input_ids, mask, src_key_padding_mask=padding_mask)
        logits = self.prob_linear(output)
        return logits      
    
    def generate(self, input_ids, eos ,pad, im_end, temperature=1, top_k=1):
        seq_len = input_ids.shape[1]
        while seq_len < MAX_SEQ_LEN:
            padding_mask = torch.zeros(1, seq_len)
            logits = self.forward(input_ids ,padding_mask)
            logits = logits[0,-1,:]/temperature
            values, indices = torch.topk(logits, top_k)
            values = F.softmax(values, dim=-1)
            sum = 0
            rnd = random.random()
            token_id = -1
            for i in range(top_k):
                sum += values[i]
                if sum >= rnd: 
                    token_id = indices[i]
                    break
            if token_id == eos or token_id==pad or token_id==im_end:
                break
            input_ids = torch.cat((input_ids, torch.tensor([[token_id]])), 1)
            seq_len += 1
        return input_ids.squeeze(0).tolist()[1:]

if __name__ == "__main__":
    chatgpt = GPT(VOCAB_SIZE, GPT_DIM, MAX_SEQ_LEN, GPT_HEAD, GPT_FF, GPT_BLOCKS)
    input_ids = torch.randint(500, VOCAB_SIZE, (1,5))
    padding_mask = torch.zeros(1,5)
    # logits = chatgpt(input_ids, padding_mask)
    # print(logits.shape)

    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer.pkl')
    eos_id = tokenizer.encode(EOS)[0]
    pad_id = tokenizer.encode(PAD)[0]
    im_end_id = tokenizer.encode(IM_END)[0]
    output_ids = chatgpt.generate(input_ids, eos_id, pad_id, im_end_id)
    output = tokenizer.decode(output_ids)
    print(f"output:\n{output}")