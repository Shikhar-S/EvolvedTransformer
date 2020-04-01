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
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,ff_hidden*d_model),
            nn.ReLU(),
            nn.Linear(ff_hidden*d_model,d_model),
        )

    def forward(self,x):
        normed = self.layer_norms[0](x)
        attended,_ = self.attentions[0](normed,normed,normed,need_weights=False)
        normed = self.layer_norms[1](attended+x)
        forwarded = self.feed_forward(normed)
        
        normed = self.layer_norms[2](forwarded+attended)
        attended,_ = self.attentions[1](normed,normed,normed,need_weights=False)
        normed = self.layer_norms[3](forwarded+attended)
        forwarded = self.feed_forward(normed)
        return forwarded+attended