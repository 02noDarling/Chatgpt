from config import *
from gpt import GPT
from dataset import GPTDataset, my_collate_fn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F
import torch.nn as nn
import torch

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
    dataset = GPTDataset(text_list)
    print(f'共加载{len(dataset)}条数据')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn)

    chatgpt = GPT(VOCAB_SIZE, GPT_DIM, MAX_SEQ_LEN, GPT_HEAD, GPT_FF, GPT_BLOCKS)
    optimizer = optim.Adam(chatgpt.parameters(), lr=1e-5)  # Adam优化器

    try:
        checkpoints = torch.load('checkpoints.pth')
        chatgpt.load_state_dict(checkpoints['model'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        iter = checkpoints['iter']
        print(f"iter:{iter}")
    except:
        iter = 0
 
    for epoch in tqdm(range(iter,TRAIN_ITER), desc="Training progress", unit="epoch"):
        for batch in tqdm(dataloader, desc="Training batch", unit="batch"):
            input_ids, padding_mask = batch
            logits = chatgpt(input_ids, padding_mask)
            pred = logits[:,:-1,:]
            target = input_ids[:,1:]
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(pred.reshape(-1,pred.shape[-1]),target.reshape(-1))
            loss = loss.reshape(-1, target.shape[-1])
            padding_mask = padding_mask[:,1:]
            loss *= 1-padding_mask
            mean_loss = loss.sum()/(1-padding_mask).sum()
            
            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()
            print(f"mean_loss:{mean_loss}")

        if epoch % 10 ==0:
            checkpoints = {'iter':epoch,'model':chatgpt.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(checkpoints,'checkpoints.pth')
            print("权重已经保存成功")

