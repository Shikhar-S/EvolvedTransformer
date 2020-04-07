import torchtext
from torchtext.datasets import text_classification
from classification_transformer import ClassificationTransformer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
import utils
from data_reader import DataReader

logger=utils.get_logger()

def generate_batch(batch,MAX_SEQ_LEN,data_reader):
    label = torch.tensor([data_reader.get_label_id(entry.Label) for entry in batch])
    text = [data_reader.get_token_tensor(entry.Text,MAX_SEQ_LEN) for entry in batch]
    text = [torch.unsqueeze(entry,0) for entry in text]
    text = torch.cat(text)
    return text, label

def train_func(sub_train_,data_reader,model,BATCH_SIZE,device,optimizer,scheduler,criterion,MAX_SEQ_LEN):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,collate_fn=lambda b: generate_batch(b,MAX_SEQ_LEN,data_reader))
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

def test(data_,data_reader,model,BATCH_SIZE,device,optimizer,criterion,MAX_SEQ_LEN):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=lambda b: generate_batch(b,MAX_SEQ_LEN,data_reader))
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
    EVOLVED = args.evolved
    #logging config vars
    logger.info('Device:%s|Batch size:%s|EmbedDim:%s|Epochs:%s|Ngrams:%s|MAX_LEN:%s|Split:%s',device.type,BATCH_SIZE,EMBED_DIM,N_EPOCHS,NGRAMS,MAX_SEQ_LEN,TRAIN_SPLIT)
    
    import os
    #download data
    logger.info('Loading Data')
    # train_dataset,test_dataset,x_vocab,y_vocab=read_data('./.data/ag_news_csv',ngrams=NGRAMS)
    data_reader = DataReader('./.data/ag_news_csv')
    #train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='./.data', ngrams=NGRAMS, vocab=None)
    logger.info('Data Loaded')
    
    VOCAB_SIZE = data_reader.get_vocab_size()
    NUM_CLASS = data_reader.get_num_classes()

    train_dataset = data_reader.get_training_data()
    test_dataset = data_reader.get_testing_data()

    model = ClassificationTransformer(EMBED_DIM,VOCAB_SIZE,NUM_CLASS,max_seq_len=MAX_SEQ_LEN,evolved=EVOLVED)
    model.to(device)

    from torch.utils.data.dataset import random_split
    min_valid_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(train_dataset) * TRAIN_SPLIT)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_func(sub_train_,data_reader,model,BATCH_SIZE,device,optimizer,scheduler,criterion,MAX_SEQ_LEN)
        logger.info('Trained for epoch %s',str(epoch))
        valid_loss, valid_acc = test(sub_valid_,data_reader,model,BATCH_SIZE,device,optimizer,criterion,MAX_SEQ_LEN)
        
        TrainingLossStr=f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)'
        ValidationLossStr=f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)'
        logger.info(TrainingLossStr)
        logger.info(ValidationLossStr)

if __name__ == "__main__":
    args,unparsed = config.get_args()
    if len(unparsed)>0:
        logger.warning('Unparsed args: %s',unparsed)
    main(args)

    