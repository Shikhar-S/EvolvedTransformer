import pandas as pd
from torchtext import data
import torch
import torch.nn.functional as F

class DataReader:
    def __init__(self,base_path):
        super(DataReader,self).__init__()
        self.TEXT_FIELD = data.Field(lower=True,tokenize='spacy',batch_first=True)
        self.LABEL_FIELD = data.Field(batch_first=True)

        self.train,self.test=data.TabularDataset.splits(path=base_path, train='train.csv', test='test.csv', format='csv',fields=[('Label', self.LABEL_FIELD),('Text', self.TEXT_FIELD)])
        self.TEXT_FIELD.build_vocab(self.train)
        self.LABEL_FIELD.build_vocab(self.train)

    def get_vocab_size(self):
        return len(self.TEXT_FIELD.vocab)

    def get_num_classes(self):
        return len(self.LABEL_FIELD.vocab)

    def get_training_data(self):
        return self.train
    
    def get_testing_data(self):
        return self.test

    def get_token_tensor(self,text,MAX_SEQ_LEN):
        ret = torch.tensor([self.TEXT_FIELD.vocab.stoi[s] for s in text])
        ret = F.pad(ret, (0,MAX_SEQ_LEN-ret.shape[0]), mode='constant', value=0)
        return ret

    def get_label_id(self,label_text):
        return self.LABEL_FIELD.vocab.stoi[label_text[0]]