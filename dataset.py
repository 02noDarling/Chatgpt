from tokenizer import BPETokenizer
from torch.utils.data import Dataset, DataLoader
from config import *
import copy
import torch
import json

tokenizer = BPETokenizer()
tokenizer.load('tokenizer.pkl')
pad_id = tokenizer.encode(PAD)[0]

class GPTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]
        text = BOS + text + EOS
        return tokenizer.encode(text)

def my_collate_fn(batch):
    max_len = 0
    for token_list in batch:
        max_len = max(max_len,len(token_list))
    for i in range(len(batch)):
        token_list = copy.deepcopy(batch[i])
        for j in range(max_len-len(token_list)):
            batch[i].append(pad_id)
    return torch.tensor(batch)

if __name__ == "__main__":
    with open('纳兰性德诗集.json','r',encoding='utf-8') as fp:
        ds=json.loads(fp.read())

    text_list=[]
    sample_count=0
    for sample in ds:
        text = sample['title']
        for p in sample['para']: 
            text += "\n"+p
        sample_count+=1
        text_list.append(text)
    print('共加载%d条数据'%sample_count)

    dataset = GPTDataset(text_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn)

    for batch in dataloader:
        print(batch)
