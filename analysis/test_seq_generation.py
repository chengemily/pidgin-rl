from itertools import chain
import json
import argparse
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V

from datasets import *
from decoder import *
from encoder_v2 import *
from train_encoder_v2 import *



def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Sequence Generator')
    parser.add_argument('--model_path', type=str, default='saved_models/en_encoder/model.pt')

    parser.add_argument('--dataset_path', type=str, default='../generate-data/data/train/en.csv',
                        help='Dataset path')
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--embeds_path', type=str, default='../tokenizer/data/indexed_data_words.json',
                        help='Embeddings path')
    parser.add_argument('--vocab_path', type=str, default='../tokenizer/data/vocab_words.json',
                        help='Embeddings path')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--cuda', action='store_false',
                        help='[DONT] use CUDA')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    return parser




def main():
    args = make_parser().parse_args()
    print("[Model hyperparams]: {}".format(str(args)))

    cuda = torch.cuda.is_available() and args.cuda
    print(f'Cuda available? {torch.cuda.is_available()}')
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    seed_everything(seed=1337, cuda=cuda)

    # get ix_to_word map
    ix_to_word = create_ix_to_vocab_map(args.vocab_path)

    # Load dataset iterators
    iters = load_data(args.dataset_path, args.embeds_path, args.lang, args.batch_size, device)
    print('Finished loading data')

    # Some datasets just have the train & test sets, so we just pretend test is valid
    if len(iters) == 4:
        X_str, train_iter, val_iter, test_iter = iters
    else:
        X_str, train_iter, test_iter = iters
        val_iter = test_iter

    # get length of a sentence
    target_length = len(train_iter[0][0][0])  # TODO - double check this is the right length
    print(f'target length: {target_length}')

    # get size of vocab
    vocab = load_json(args.vocab_path)
    output_dims = len(vocab)

    print("[Corpus]: train: {}, test: {}".format(
        len(train_iter[0]) * len(train_iter[0][0]), len(test_iter[0]) * len(test_iter[0][0])))


    # load model
    model = torch.load(args.model_path, map_location=device)
    # emb = list(model.children())[:-1][0]
    # print(emb(torch.tensor(5, dtype=torch.long, device=device)))
    # input()

    # Define and compute loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # loss = evaluate_encoder(model, test_iter, criterion, device, args, type='Test') 
    # print("Test loss: ", loss)

    # make predictions with model
    ix = np.random.randint(len(test_iter))
    y = test_iter[0][ix] # actual data
    x = test_iter[2][ix].float()
    pred = model(x)


    # Print some sample evaluations
    translated_batch = translate_batch(pred, ix_to_word)
    translated_y = translate_batch(y, ix_to_word)
    for ix, v in enumerate(x[:10]):
        print(f'\nOriginal instructions: \n{v}')
        print(f'Predicted:\n{translated_batch[ix]}')
        print(f'Actual:\n{translated_y[ix]}')


if __name__ == '__main__':
    main()







