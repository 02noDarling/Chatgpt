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

    chatgpt = GPT()
    checkpoints = torch.load('checkpoints.pth')
    chatgpt.load_state_dict(checkpoints['model'])
    print(f"Total parameters: {count_parameters(chatgpt)}")
    for param_tensor in chatgpt.state_dict():
        #打印 key value字典
        print(param_tensor,'\t',chatgpt.state_dict()[param_tensor].size())
    
    images = torch.randn(10, 3, IMAGE_SIZE, IMAGE_SIZE)
    text = "春日"
    text = BOS + text
    input_ids = tokenizer.encode(text)
    input_ids =torch.tensor(input_ids).unsqueeze(0)
    output_ids = chatgpt.generate(input_ids, eos_id, pad_id, im_end_id, images)
    output = tokenizer.decode(output_ids)
    print(f"output:\n{output}")
    