VOCAB_SIZE=2000    # 词表大小
MAX_SEQ_LEN=500     # GPT模型输入限制
DIM=384

# gpt transformer
GPT_HEAD=6
GPT_FF=1024
GPT_BLOCKS=6

# vision transformer
VIT_OUT_CHANNEL = 100
VIT_HEAD=6
VIT_FF=1024
VIT_BLOCKS=6

#image
IMAGE_SIZE = 224
PATCH_SIZE =16

# training
TRAIN_ITER=10000
BATCH_SIZE=1

# inference
TEMPERATURE=1.2
TOP_K=1

# special tokens
BOS='<|beginoftext|>'
EOS='<|endoftext|>'
PAD='<|padding|>'
IM_START='<|im_start|>'
IM_END='<|im_end|>'

# chat or generate
GPT_MODE='generate'