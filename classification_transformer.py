from transformer_block import TransformerBlock
from evolved_transformer_block import EvolvedTransformerBlock
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from embedder import Embedder

class ClassificationTransformer(nn.Module):
    def __init__(self,d_model,vocab_size,num_classes,num_heads=8,max_seq_len =200,dropout=0.1,max_pool=True,evolved=False):
        super(ClassificationTransformer,self).__init__()
        self.max_pool=max_pool
        self.embedder = Embedder(vocab_size,d_model,dropout,max_seq_len)
        self.transformer_block = EvolvedTransformerBlock(d_model,num_heads) if evolved else TransformerBlock(d_model,num_heads) 
        self.to_probability = nn.Linear(d_model,num_classes)
        
    def forward(self,x):
        x=self.embedder(x)
        x=self.transformer_block(x)
        x=self.to_probability(x)
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension
        return F.log_softmax(x,dim=1)

