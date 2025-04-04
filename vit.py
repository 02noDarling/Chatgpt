import torch
import torch.nn as nn
from config import *

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.patch_linear = nn.Linear(VIT_OUT_CHANNEL, DIM)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, DIM))
        self.cls_token = nn.Parameter(torch.randn(1, 1, DIM))
        self.conv = nn.Conv2d(in_channels=3, out_channels=VIT_OUT_CHANNEL, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(DIM, VIT_HEAD, VIT_FF, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, VIT_BLOCKS)

    def forward(self, images):
        # images shape:[batch, channel=3, width=224, height=224]
        images = self.conv(images) # images shape:[batch, channel=VIT_OUT_CHANNEL, width=224/16=14, height=14]
        images = images.reshape(images.shape[0],images.shape[1], -1) # images shape:[batch, channel=VIT_OUT_CHANNEL, seq_len=14*14]
        images = torch.transpose(images, -2, -1) #images shape:[batch, seq_len=14*14, emb_size=VIT_OUT_CHANNEL]
        patches = self.patch_linear(images) #patches shape:[batch, seq_len=14*14, emb_size=dim]
        patches = torch.cat((patches, self.cls_token.expand(patches.shape[0], -1, -1)), dim=1)
        patches += self.position_embedding.expand(patches.shape[0], -1, -1)
        return self.encoder(patches) 

if __name__ == "__main__":
    batch_size = 10
    vit = VisionTransformer()
    images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    output = vit(images)
    print(output.shape)
