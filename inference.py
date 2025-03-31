from gpt import GPT
from tokenizer import BPETokenizer
from config import *
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer.pkl')
    eos_id = tokenizer.encode(EOS)[0]
    pad_id = tokenizer.encode(PAD)[0]
    im_end_id = tokenizer.encode(IM_END)[0]

    chatgpt = GPT(VOCAB_SIZE, GPT_DIM, MAX_SEQ_LEN, GPT_HEAD, GPT_FF, GPT_BLOCKS)
    checkpoints = torch.load('checkpoints.pth')
    chatgpt.load_state_dict(checkpoints['model'])
    print(f"Total parameters: {count_parameters(chatgpt)}")

    text = "春日"
    text = BOS + text
    input_ids = tokenizer.encode(text)
    input_ids =torch.tensor(input_ids).unsqueeze(0)
    output_ids = chatgpt.generate(input_ids, eos_id, pad_id, im_end_id)
    output = tokenizer.decode(output_ids)
    print(f"output:\n{output}")
    