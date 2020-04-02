import argparse
import logging
import utils
logger = utils.get_logger()

def str2bool(v):
    return v.loner() in ('true')

parser = argparse.ArgumentParser()
parser.add_argument("--batch",type=int,default=16)
parser.add_argument("--evolved",type=str2bool,default=False)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--model_dim",type=int,default=32)
parser.add_argument("--max_seq_len",type=int,default=200)
parser.add_argument("--backend",type=str,default='auto',choices=['cpu', 'gpu','auto'])
parser.add_argument('--ngrams',type=int,default=2)
parser.add_argument('--train_split',type=float,default=0.95)

def get_args():
    logger.info('Parsing arguments')
    args,unparsed = parser.parse_known_args()
    return args, unparsed