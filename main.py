import torchtext
from torchtext.datasets import text_classification
from classification_transformer import ClassificationTransformer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
import utils

logger=utils.get_logger()

import torch.nn.functional as F

def generate_batch(batch,MAX_SEQ_LEN):
    label = torch.tensor([entry[0] for entry in batch])
    text=[F.pad(entry[1], (0,MAX_SEQ_LEN-entry[1].shape[0]), mode='constant', value=0) for entry in batch]
    text=[torch.unsqueeze(entry,0) for entry in text]
    text = torch.cat(text)
    return text, label

def train_func(sub_train_,model,BATCH_SIZE,device,optimizer,scheduler,criterion,MAX_SEQ_LEN):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,collate_fn=lambda b: generate_batch(b,MAX_SEQ_LEN))
    for i, (text, category) in enumerate(tqdm(data)):
        optimizer.zero_grad()
        text, category = text.to(device), category.to(device)
        output = model(text)
        loss = criterion(output, category)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == category).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_,model,BATCH_SIZE,device,optimizer,criterion,MAX_SEQ_LEN):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=lambda b: generate_batch(b,MAX_SEQ_LEN))
    for text, category in tqdm(data):
        text, category = text.to(device), category.to(device)
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, category)
            loss += loss.item()
            acc += (output.argmax(1) == category).sum().item()

    return loss / len(data_), acc / len(data_)

def main(args):
    #Initialise config vars
    NGRAMS=args.ngrams
    BATCH_SIZE=args.batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(args.backend == 'gpu' and not torch.cuda.is_available()):
        logger.error('Backend device: %s not available',args.backend)
    if args.backend != 'auto':
        device = torch.device('cpu'  if args.backend=='cpu' else 'cuda')
    EMBED_DIM=args.model_dim
    N_EPOCHS=args.epochs
    MAX_SEQ_LEN=args.max_seq_len
    TRAIN_SPLIT=args.train_split
    #logging config vars
    logger.info('Device:%s|Batch size:%s|EmbedDim:%s|Epochs:%s|Ngrams:%s|MAX_LEN:%s|Split:%s',device.type,BATCH_SIZE,EMBED_DIM,N_EPOCHS,NGRAMS,MAX_SEQ_LEN,TRAIN_SPLIT)
    
    import os
    #download data
    logger.info('Loading Data')
    if not os.path.isdir('./.data'):
        os.mkdir('./.data')
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='./.data', ngrams=NGRAMS, vocab=None)
    logger.info('Data Loaded')

    VOCAB_SIZE = len(train_dataset.get_vocab())
    NUM_CLASS = len(train_dataset.get_labels())
    model = ClassificationTransformer(EMBED_DIM,VOCAB_SIZE,NUM_CLASS,max_seq_len=MAX_SEQ_LEN)
    model.to(device)

    from torch.utils.data.dataset import random_split
    min_valid_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(train_dataset) * TRAIN_SPLIT)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_func(sub_train_,model,BATCH_SIZE,device,optimizer,scheduler,criterion,MAX_SEQ_LEN)
        logger.info('Trained for epoch %s',str(epoch))
        valid_loss, valid_acc = test(sub_valid_,model,BATCH_SIZE,device,optimizer,criterion,MAX_SEQ_LEN)
        
        TrainingLossStr=f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)'
        ValidationLossStr=f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)'
        logger.info(TrainingLossStr)
        logger.info(ValidationLossStr)

if __name__ == "__main__":
    args,unparsed = config.get_args()
    if len(unparsed)>0:
        logger.warning('Unparsed args: %s',unparsed)
    main(args)

    