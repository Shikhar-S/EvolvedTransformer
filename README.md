# Transformer Encoder block implementation in PyTorch

Pytorch implementations of:

* Transformer Encoder block from [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
* Evolved Transformer Encoder block from [Evolved Transformer](https://arxiv.org/abs/1901.11117).
* Gated Linear Unit from [Language Modelling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083).

for text classification on AG_NEWS dataset.

<p align="center"><img src="/assets/EvolvedTransformer.png" alt="TransformerModels"></p>

## Prerequisites
- numpy==1.17.2
- torch==1.2.0
- torchtext==0.5.0
- tqdm==4.44.1
- spacy==2.2.4
- Python 3.6+
 
## Usage

Install prerequisites with:
    
    pip3 install -r requirements.txt
    
    python3 -m spacy download en

To run **Encoder** for text classification on AG_NEWS dataset:

    python3 main.py 

    python3 main.py --evolved true 

More configurations can be found [here](config.py).

## Reference

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [Evolved Transformer](https://arxiv.org/abs/1901.11117).
- [Language Modelling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083).


## Author

Shikhar / [@Shikhar](https://shikhar-s.github.io)
