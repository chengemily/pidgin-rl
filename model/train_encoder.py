import argparse
import json
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn

from datasets import dataset_map
from model import *


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
    parser.add_argument('--dataset_path', type=str, default='',
                        help='Dataset path')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
    parser.add_argument('--hidden', type=int, default=300, # changing hidden to match emsize
                        help='number of hidden units for the RNN decoder')
    parser.add_argument('--nlayers', type=int, default=2,
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
    parser.add_argument('--fine', action='store_true',
                        help='use fine grained labels in SST')
    return parser


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


SOS_token = 1
EOS_token = 2

def train_encoder(model, data, optimizer, criterion, args):
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
    total_loss = 0

    for batch_num, batch in enumerate(data):
        model.zero_grad()
        x, lens = batch.text # what is lens here?
        y = batch.label

        # Forward pass
        pred = model(x) # pred will return a tensor of embbed vectors, representing words

        # Compute loss
        loss = criterion(pred, y) # make sure y is off
        total_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        print("[Batch]: {}/{} in {:.5f} seconds. Loss: {}".format(
            batch_num, len(data), time.time() - t, total_loss / (batch_num * len(batch))), end='\r', flush=True)
        t = time.time()

    print()
    print("[Loss]: {:.5f}".format(total_loss / len(data)))
    return total_loss / len(data)


def evaluate(model, data, criterion, type='Valid'):
    model.eval()
    t = time.time()
    total_loss = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, lens = batch.text
            y = batch.label

            pred, attn = model(x)
            total_loss += float(criterion(pred, y))
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data), time.time() - t), end='\r', flush=True)
            t = time.time()

    print()
    print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
    print(attn)
    return total_loss / len(data)


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
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    seed_everything(seed=1337, cuda=cuda)
    vectors = load_pretrained_vectors(args.dataset_path)

    # Load dataset iterators
    # TODO figure out data loading
    # iters, TEXT, LABEL = dataset_map[args.data](args.batch_size, device=device, vectors=vectors)

    # Some datasets just have the train & test sets, so we just pretend test is valid
    if len(iters) == 3:
        train_iter, val_iter, test_iter = iters
    else:
        train_iter, test_iter = iters
        val_iter = test_iter

    print("[Corpus]: train: {}, test: {}, vocab: {}, labels: {}".format(
        len(train_iter.dataset), len(test_iter.dataset), len(TEXT.vocab), len(LABEL.vocab)))

    embedding = nn.Embedding(ntokens, args.emsize, padding_idx=1, max_norm=1) # TODO - padding_idx = 0?
    # if vectors: embedding.weight.data.copy_(TEXT.vocab.vectors)

    # Define model pipeline
    fc_layer_dims = [args.hidden] #output of FC should be h0, first hidden input
    decoder = Decoder(args.emsize, args.hidden, rnn_type=args.model, nlayers=args.nlayers,
                      dropout=args.drop, bidirectional=args.bi)

    attention_dim = args.hidden if not args.bi else 2 * args.hidden
    attention = Attention(attention_dim, attention_dim, attention_dim)

    model = SequenceGenerator(embedding, decoder, fc_layer_dims, attention)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True)

    # Train and validate per epoch
    try:
        best_valid_loss = None

        for epoch in range(1, args.epochs + 1):
            train(model, train_iter, optimizer, criterion, args)
            loss = evaluate(model, val_iter, optimizer, criterion, args)

            if not best_valid_loss or loss < best_valid_loss:
                best_valid_loss = loss

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    loss = evaluate(model, test_iter, optimizer, criterion, args, type='Test')
    print("Test loss: ", loss)


if __name__ == '__main__':
    main()
