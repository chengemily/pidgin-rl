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


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Sequence Generator')
    parser.add_argument('--save_path', type=str, default='saved_models/fr_encoder/model.pt')
    parser.add_argument('--dataset_path', type=str, default='../generate-data/data/train/fr.csv',
                        help='Dataset path')
    parser.add_argument('--lang', type=str, default='fr')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
    parser.add_argument('--embeds_path', type=str, default='../tokenizer/data/indexed_data.json',
                        help='Embeddings path')
    parser.add_argument('--vocab_path', type=str, default='../tokenizer/data/vocab.json',
                        help='Embeddings path')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
    parser.add_argument('--hidden', type=int, default=300,  # changing hidden to match emsize
                        help='number of hidden units for the RNN decoder')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers of the RNN decoder')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--drop', type=float, default=0,
                        help='dropout')
    parser.add_argument('--bi', action='store_true',
                        help='[USE] bidirectional encoder')
    parser.add_argument('--cuda', action='store_false',
                        help='[DONT] use CUDA')

    return parser


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) != tuple:  # if the hidden state is a variable (see imports)
        return h.detach()  # detach from history
    else:
        return tuple(repackage_hidden(v) for v in h)


def train_encoder(model, data, optimizer, criterion, device, args):
    """
    :param model: (nn.Module) model
    :param data: iterator of data
    :param optimizer:
    :param criterion: loss function(pred, actual)
    :param args:
    :return:
    """
    model.train()
    t = time.time()

    total_loss = 0  # is this redundant with total_loss defined? -> this one will store criterion, total_loss stores cumulative loss

    for batch_num, batch in enumerate(data[0]):
        model.zero_grad()
        # x is coordinate, y is output indices
        # print(f'batch size: {len(batch)}')

        x = data[2][batch_num].float()  # .to(device) # make the coordinates the predictors x
        y = batch  # .to(device) # label (indices) is the word embeddings
        # print(f'initial y:{y}')
        #  print(f'initial y shape : {y.size()}')
        # Forward pass
        # print(f'initial x shape : {x.size()}')
        # print(f'initial x  : {x}')

        with torch.autograd.set_detect_anomaly(True):
            # forward pass
            # Forward pass
            pred = model(x.to(device))

            # if batch_num % 100 == 0:
            #     print(pred)
            #     print(y)

            # Compute loss
            loss = criterion(pred, y) # TODO - pred might be the wrong dimensions (switch 1 and 2)
            total_loss += loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            # Detach pred?
            pred.detach()

            print("[Batch]: {}/{} in {:.5f} seconds. Loss: {}".format(
                batch_num, len(data[0]), time.time() - t, loss / (batch_num * len(batch))), end='\r', flush=True)
            t = time.time()

    print()
    print("[Loss]: {:.5f}".format(loss / len(data)))
    return loss / (args.batch_size * len(data[0]))



def evaluate_encoder(model, data, criterion, args, type='Valid'):
    model.eval()
    t = time.time()
    total_loss = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data[0]):
            x = data[2][batch_num].float()  # .to(device) # make the coordinates the predictors x
            y = batch

            pred = model(x.to(args.device))
            total_loss += float(criterion(pred, y))
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data[0]), time.time() - t), end='\r', flush=True)
            t = time.time()

            if batch_num == 1:
                print(pred)
                print(y)

    print()
    print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
    return total_loss / (len(data[0]) * args.batch_size)


def load_pretrained_vectors(vectors):
    """
    Load pretrained embeddings as dict
    :param vectors_path: (str)
    :return:
    """
    with open(vectors) as f:
        data = json.load(f)
    return data


def main():
    args = make_parser().parse_args()
    print("[Model hyperparams]: {}".format(str(args)))

    cuda = torch.cuda.is_available() and args.cuda
    print(f'Cuda available? {torch.cuda.is_available()}')
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    seed_everything(seed=1337, cuda=cuda)

    # Load dataset iterators
    # TODO figure out data loading
    # iters, TEXT, LABEL = dataset_map[args.data](args.batch_size, device=device, vectors=vectors)
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

    # Load or define embedding layer
    # if args.use_pretrained:
    #     embedding = nn.Linear(args.emsize, args.emsize)
    # else:
    #     print(f'initialize new embedding: {len(vocab)}')
    #     embedding = nn.Embedding(len(vocab), args.emsize, padding_idx=0)
    embedding = nn.Embedding(args.emsize, args.hidden)  # 1st param - size of vocab, 2nd param - size of embedding vector

    # Define model pipeline
    # FCL
    fc_layer_dims = [args.hidden]  # output of FC should be h0, first hidden input

    # RNN
    decoder = Decoder(output_dims, args.hidden, embedding, rnn_type=args.model, nlayers=args.nlayers,
                      dropout=args.drop)  # TODO - more thoroughly check this

    # Sequence Generator
    sequence_gen = Sequence_Generator(embedding,
                                      decoder,
                                      fc_layer_dims,
                                      args.hidden,
                                      target_length,
                                      output_dims,
                                      args.batch_size,
                                      output_dims,
                                      rnn_type=args.model,
                                      device=device)


    # put all models on GPU
    sequence_gen.to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(sequence_gen.paramteres(), args.lr, amsgrad=True)

    # Train and validate per epoch
    try:
        best_valid_loss = None

        for epoch in range(1, args.epochs + 1):
            print(f'Epoch: {epoch}')
            train_encoder(sequence_gen, train_iter, optimizer, criterion, device, args)
            loss = evaluate_encoder(sequence_gen, val_iter, criterion, args)

            if not best_valid_loss or loss < best_valid_loss:
                best_valid_loss = loss

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    loss = evaluate_encoder(model, test_iter, optimizer, criterion, args, type='Test')
    print("Test loss: ", loss)

    # Save model
    torch.save(sequence_gen, args.save_path)

    # Print some sample evaluations
    batch_to_print_X, batch_to_print_X_lens, batch_to_print_Y = val_iter[0][0]
    pred = model(batch_to_print_Y)
    print("Predictions: ", pred)
    print("Original: ", batch_to_print_Y)


if __name__ == '__main__':
    main()







