import torch
import torch.nn as nn
from embedding import EmbeddingWithPosition
from config import *
import torch.nn.functional as F
import random
from tokenizer import BPETokenizer
from vit import VisionTransformer

class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.embedding = EmbeddingWithPosition()

        # 用encoder加掩码的方式实现decoder-only
        decoder_layer = nn.TransformerEncoderLayer(DIM, GPT_HEAD, GPT_FF, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, GPT_BLOCKS)
        self.prob_linear = nn.Linear(DIM, VOCAB_SIZE)
        self.vit = VisionTransformer()
    
    def get_vlm_mask(self, images_len, seq_len):
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(images_len):
            for j in range(images_len):
                mask[i][j] = 0
        for i in range(images_len, seq_len):
            for j in range(i):
                mask[i][j] = 0
        return mask

    def forward(self, input_ids, padding_mask=None, images=None): # x: [batch, seq_len]
        input_ids = self.embedding(input_ids)
        text_len = input_ids.shape[1]
        if images != None:
            images = self.vit(images)
            seq_len = images.shape[1] + input_ids.shape[1]
            mask = self.get_vlm_mask(images.shape[1], seq_len)
            images_padding_mask = torch.zeros(images.shape[0], images.shape[1])
            padding_mask = torch.cat((images_padding_mask, padding_mask), dim=-1)
            input_ids = torch.cat((images, input_ids), dim=1)
        else: 
            seq_len = input_ids.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        output = self.decoder(input_ids, mask, src_key_padding_mask=padding_mask)
        logits = self.prob_linear(output)
        return logits[:,-text_len:,:]    
    
    def generate(self, input_ids, eos ,pad, im_end, temperature=1, top_k=1, images=None):
        if images != None:
            seq_len = input_ids.shape[1] + images.shape[1]
        else:
            seq_len = input_ids.shape[1]
        while seq_len < MAX_SEQ_LEN:
            padding_mask = torch.zeros(1, input_ids.shape[1])
            logits = self.forward(input_ids ,padding_mask, images)
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
    chatgpt = GPT()
    # input_ids = torch.randint(500, VOCAB_SIZE, (1,5))
    # padding_mask = torch.zeros(1,5)
    input_ids = torch.randint(500, VOCAB_SIZE, (10, 5))
    padding_mask = torch.zeros(10, 5)
    images = torch.randn(10, 3, IMAGE_SIZE, IMAGE_SIZE)
    logits = chatgpt(input_ids, padding_mask, images)
    print(logits.shape)


    input_ids = torch.randint(500, VOCAB_SIZE, (1,5))
    padding_mask = torch.zeros(1,5)
    images = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer.pkl')
    eos_id = tokenizer.encode(EOS)[0]
    pad_id = tokenizer.encode(PAD)[0]
    im_end_id = tokenizer.encode(IM_END)[0]
    output_ids = chatgpt.generate(input_ids, eos_id, pad_id, im_end_id, images=images)
    output = tokenizer.decode(output_ids)
    print(f"output:\n{output}")