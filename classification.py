import torchtext
from torchtext.datasets import text_classification
from classification_transformer import ClassificationTransformer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

NGRAMS = 2
import os
#download data
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='./.data', ngrams=NGRAMS, vocab=None)

BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
MAX_SEQ_LEN=200
NUM_CLASS = len(train_dataset.get_labels())
model = ClassificationTransformer(EMBED_DIM,VOCAB_SIZE,NUM_CLASS,max_seq_len=MAX_SEQ_LEN)
model.to(device)

import torch.nn.functional as F
def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text=[F.pad(entry[1], (0,200-entry[1].shape[0]), mode='constant', value=0) for entry in batch]
    text=[torch.unsqueeze(entry,0) for entry in text]
    text = torch.cat(text)
    return text, label

def train_func(sub_train_):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,collate_fn=generate_batch)
    for i, (text, category) in enumerate(tqdm(data)):
        optimizer.zero_grad()
        text, category = text.to(device), category.to(device)
        print(text.device)
        print(category.device)
        for param in model.parameters():
            print(param.is_cuda)
        output = model(text)
        loss = criterion(output, category)
        train_loss += loss.item()
        print(loss.item())
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == category).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, category in tqdm(data):
        text, category = text.to(device), category.to(device)
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, category)
            loss += loss.item()
            acc += (output.argmax(1) == category).sum().item()

    return loss / len(data_), acc / len(data_)

import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    print('Trained for epoch',epoch,', now validating.')
    valid_loss, valid_acc = test(sub_valid_)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')