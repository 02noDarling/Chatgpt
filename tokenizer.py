import re
import pickle
from tqdm import tqdm

class BPETokenizer:
    def __init__(self):
        self.byte2id = {}
        self.id2byte = {}
        self.next_id = 0
        
        # 特殊token
        self.sp_str2id = {}
        self.sp_id2str = {}
    
    def add_special_tokens(self, special_token_list):
        for token in special_token_list:
            if not token in self.sp_str2id:
                self.sp_str2id[token] = self.next_id
                self.sp_id2str[self.next_id] = token
                self.next_id += 1
    
    def train(self, text_lists, vocab_size):
        for i in range(256):
            self.byte2id[bytes([i])] = self.next_id
            self.id2byte[self.next_id] = bytes([i])
            self.next_id += 1

        bytes_lists = []
        for text in text_lists:
            bytes_list = []
            for b in text.encode('utf-8'):
                bytes_list.append(bytes([b]))
            if len(bytes_list) != 0:
                bytes_lists.append(bytes_list)
        progress=tqdm(total=vocab_size,initial=self.next_id)

        while self.next_id < vocab_size:
            bytes_map = {}
            chosen_bytes = None
            max_counts = 0
            for bytes_list in bytes_lists:
                for i in range(1,len(bytes_list)):
                    if not bytes_list[i-1]+bytes_list[i] in bytes_map:
                        bytes_map[bytes_list[i-1]+bytes_list[i]] = 0 
                    else:
                        bytes_map[bytes_list[i-1]+bytes_list[i]] += 1
                    if bytes_map[bytes_list[i-1]+bytes_list[i]] > max_counts:
                        max_counts = bytes_map[bytes_list[i-1]+bytes_list[i]]
                        chosen_bytes =  bytes_list[i-1]+bytes_list[i]

            if chosen_bytes is None:
                break
            
            self.byte2id[chosen_bytes] = self.next_id
            self.id2byte[self.next_id] = chosen_bytes
            self.next_id += 1
            progress.update(1)
            
            for i in range(len(bytes_lists)):
                bytes_list = bytes_lists[i]
                new_bytes_lists = []
                pre_bytes = bytes_list[0]
                for j in range(1,len(bytes_list)):
                    if pre_bytes ==None:
                        pre_bytes = bytes_list[j]
                        continue
                    if pre_bytes + bytes_list[j] == chosen_bytes:
                        new_bytes_lists.append(chosen_bytes)
                        pre_bytes = None
                    else:
                        new_bytes_lists.append(pre_bytes)
                        pre_bytes = bytes_list[j]
                if pre_bytes != None:
                    new_bytes_lists.append(pre_bytes)
                bytes_lists[i] = new_bytes_lists
                

    def encode(self, text):
        special_tokens = []
        for key in self.sp_str2id:
            special_tokens.append(key)
        text_list = self.split_with_special_tokens(text, special_tokens)
        token_ids = [ ]
        for text in text_list:
            if text in self.sp_str2id:
                token_ids.append(self.sp_str2id[text])
            else:
                bytes_list = []
                for b in text.encode('utf-8'):
                    bytes_list.append(bytes([b]))
                while True:
                    bytes_map = {}
                    chosen_bytes = None
                    max_counts = 0
                    min_id = len(self.byte2id) + len(self.sp_str2id) + 100
                    for i in range(1,len(bytes_list)):
                        if not bytes_list[i-1]+bytes_list[i] in bytes_map:
                            bytes_map[bytes_list[i-1]+bytes_list[i]] = 0 
                        else:
                            bytes_map[bytes_list[i-1]+bytes_list[i]] += 1
                    for key,value in bytes_map.items():
                        if not key in self.byte2id:
                            continue
                        if value > max_counts or (value==max_counts and bytes_map[key] < min_id):
                            chosen_bytes = key
                            max_counts = value
                            min_id = bytes_map[key]
                    
                    if chosen_bytes ==None:
                        break
                    
                    new_bytes_lists = []
                    pre_bytes = bytes_list[0]
                    for j in range(1,len(bytes_list)):
                        if pre_bytes ==None:
                            pre_bytes = bytes_list[j]
                            continue
                        if pre_bytes + bytes_list[j] == chosen_bytes:
                            new_bytes_lists.append(chosen_bytes)
                            pre_bytes = None
                        else:
                            new_bytes_lists.append(pre_bytes)
                            pre_bytes = bytes_list[j]
                    if pre_bytes != None:
                        new_bytes_lists.append(pre_bytes)
                    bytes_list = new_bytes_lists

                for b in bytes_list:
                    token_ids.append(self.byte2id[b])
        

        return token_ids

    def decode(self, token_ids):
        bytes_list = []
        for token_id in token_ids:
            if token_id in self.sp_id2str:
                bytes_list.append(self.sp_id2str[token_id].encode('utf-8'))
            else:
                bytes_list.append(self.id2byte[token_id])
        
        return b''.join(bytes_list).decode('utf-8',errors='replace')

    def save(self, filepath):
        # 保存tokenizer的所有参数到文件
        state = {
            'byte2id': self.byte2id,
            'id2byte': self.id2byte,
            'next_id': self.next_id,
            'sp_str2id': self.sp_str2id,
            'sp_id2str': self.sp_id2str
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Tokenizer state saved to {filepath}")
    
    def load(self, filepath):
        # 从文件加载tokenizer的参数
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.byte2id = state['byte2id']
        self.id2byte = state['id2byte']
        self.next_id = state['next_id']
        self.sp_str2id = state['sp_str2id']
        self.sp_id2str = state['sp_id2str']
        
        print(f"Tokenizer state loaded from {filepath}")

    def split_with_special_tokens(self,text, special_tokens):
        # 创建正则表达式模式，匹配特殊字符
        pattern = f"({'|'.join(map(re.escape, special_tokens))})"
        
        # 使用正则表达式进行分割，同时保留特殊字符
        result = re.split(pattern, text)

        return [x for x in result]

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer.pkl')
    print(tokenizer.next_id)
    text = "<|im_start|>我是胡戴立<|im_end|>"
    input_ids = tokenizer.encode(text)
    print(f"input_ids:\n{input_ids}")
    decode_text = tokenizer.decode(input_ids)
    print(decode_text)
