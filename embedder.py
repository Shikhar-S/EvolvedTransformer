import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self,d_model,dropout = 0.1,max_seq_len = 200):
        super(PositionalEncoder,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encodings = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term =  torch.pow(1000,2*torch.arange(0, d_model, 2).float()/d_model) 
        positional_encodings[:, 0::2] = torch.sin(position / div_term)
        positional_encodings[:, 1::2] = torch.cos(position / div_term)
        positional_encodings = positional_encodings.unsqueeze(0).transpose(0, 1) #shape is max_seq_len,1,d_model
        self.register_buffer('positional_encodings', positional_encodings)

    def forward(self, x):
        x = x + self.positional_encodings[:x.size(0), :]
        return self.dropout(x)

class Embedder(nn.Module):
    def __init__(self,vocab_size,d_model,dropout=0.1,max_seq_len=200):
        super(Embedder,self).__init__()
        if d_model%2!=0:
            d_model+=1 #ensures positional embeddings have both sine and cosine component for all indices.
        self.d_model = d_model #model embedding dimension
        self.embed = nn.Embedding(vocab_size,d_model)
        self.positional_embedder = PositionalEncoder(d_model,dropout,max_seq_len)
        
    def forward(self, x):
        embedded=self.embed(x)
        return self.positional_embedder(embedded)



