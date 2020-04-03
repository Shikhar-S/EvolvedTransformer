from embedder import Embedder, PositionalEncoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads=8,ff_hidden=4):
        super(TransformerBlock,self).__init__()
        self.attentions = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads) for _ in range(2)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.feed_forward = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model,ff_hidden*d_model),
            nn.ReLU(),
            nn.Linear(ff_hidden*d_model,d_model),
        ) for _ in range(2)])

    def forward(self,x):
        normed = self.layer_norms[0](x)
        normed=normed.transpose(0,1)
        attended,_ = self.attentions[0](normed,normed,normed,need_weights=False)
        attended=attended.transpose(0,1)
        normed = self.layer_norms[1](attended+x)
        forwarded = self.feed_forward[0](normed)
        
        normed = self.layer_norms[2](forwarded+attended)
        normed=normed.transpose(0,1)
        attended,_ = self.attentions[1](normed,normed,normed,need_weights=False)
        attended=attended.transpose(0,1)
        normed = self.layer_norms[3](forwarded+attended)
        forwarded = self.feed_forward[1](normed)
        return forwarded+attended