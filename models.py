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

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads=8,ff_hidden=4):
        super(TransformerBlock,self).__init__()
        self.attentions = [nn.MultiheadAttention(d_model, num_heads) for _ in range(2)]
        self.layer_norms = [nn.LayerNorm(d_model) for _ in range(4)]
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

class ClassificationTransformer(nn.Module):
    def __init__(self,d_model,vocab_size,num_classes,num_heads=8,max_seq_len =200,dropout=0.1,max_pool=True):
        super(ClassificationTransformer,self).__init__()
        self.max_pool=max_pool
        self.embedder = Embedder(vocab_size,d_model,dropout,max_seq_len)
        self.transformer_block = TransformerBlock(d_model,num_heads)
        self.to_probability = nn.Linear(d_model,num_classes)
        
    def forward(self,x):
        x=self.embedder(x)
        x=self.transformer_block(x)
        x=self.to_probability(x)
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension
        return F.log_softmax(x,dim=1)

