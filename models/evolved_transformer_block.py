from models.embedder import Embedder, PositionalEncoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gated_linear_unit import GLU

class EvolvedTransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads=8,ff_hidden=4):
        super(EvolvedTransformerBlock,self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads) 
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,ff_hidden*d_model),
            nn.ReLU(),
            nn.Linear(ff_hidden*d_model,d_model),
        )
        self.glu = GLU(d_model,1)
        self.left_net = nn.Sequential(
            nn.Linear(d_model,ff_hidden*d_model),
            nn.ReLU()
        )
        self.right_net = nn.Sequential(
            nn.Conv1d(in_channels=d_model,out_channels=d_model//2,kernel_size=3,padding=1),
            nn.ReLU()
        )

        self.mid_layer_norm=nn.LayerNorm(d_model*ff_hidden)
        self.sep_conv=nn.Sequential(
            nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=1,kernel_size=9,padding=4),
            nn.Conv1d(in_channels=1,out_channels=d_model,kernel_size=1)
        )

    def forward(self,x):

        glued = self.glu(self.layer_norms[0](x))+x
        glu_normed = self.layer_norms[1](glued)

        left_branch = self.left_net(glu_normed)
        right_branch = self.right_net(glu_normed.transpose(1,2)).transpose(1,2)
        right_branch = F.pad(input=right_branch, pad=(0,left_branch.shape[2]-right_branch.shape[2],0,0,0,0), mode='constant', value=0)

        mid_result = left_branch+right_branch
        mid_result = self.mid_layer_norm(mid_result)
        mid_result = self.sep_conv(mid_result.transpose(1,2)).transpose(1,2)

        mid_result = mid_result + glued

        normed = self.layer_norms[2](mid_result)
        normed=normed.transpose(0,1)
        attended = self.attention(normed,normed,normed,need_weights=False)[0].transpose(0,1) + mid_result
        normed = self.layer_norms[3](attended)
        forwarded = self.feed_forward(normed)+attended
        return forwarded