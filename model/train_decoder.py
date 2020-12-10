"""
Code based on @https://github.com/mttk/rnn-classifier
"""

import argparse
import json
import os, sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import load_data
from decoder import *


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
    parser.add_argument('--save_path', type=str, default='saved_models/en_decoder')
    parser.add_argument('--dataset_path', type=str, default='../generate-data/data/train/en.csv')
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--embeds_path', type=str, default='../tokenizer/data/indexed_data.json',
                        help='Embeddings path')
    parser.add_argument('--vocab_path', type=str, default='../tokenizer/data/vocab.json',
                        help='Embeddings path')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
    parser.add_argument('--emsize', type=int, default=20,
                        help='size of word embeddings')
    parser.add_argument('--hidden', type=int, default=50,
                        help='number of hidden units for the RNN encoder')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers of the RNN encoder')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--wdecay', type=float, default=1e-3,
                        help='weight decay')
    parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--drop', type=float, default=0.3,
                        help='dropout')
    parser.add_argument('--bi', action='store_true',
                        help='[USE] bidirectional encoder')
    parser.add_argument('--cuda', action='store_false',
                        help='[DONT] use CUDA')
    parser.add_argument('--use_outputs', action='store_true', help='concat outputs mode')
    parser.add_argument('--use_attn', action='store_true', help='use dot prod attn')
    return parser


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def train(model, data, optimizer, criterion, device, args, epoch, writer):
    """

    :param model: (nn.Module) model
    :param data: tuple of iterators of data (dataX, dataY)
    :param optimizer:
    :param criterion: loss function(pred, actual)
    :param args:
    :return:
    """
    model.train()
    t = time.time()
    total_loss = 0
    n_batches = len(data[0])
    attn_maps = []

    for batch_num, batch in enumerate(data[0]):
        model.zero_grad()

        x = batch
        x_len = data[1][batch_num]
        y = data[2][batch_num]

        # Forward pass
        pred, attn = model(x.to(device), x_len)
        attn_maps.append(attn)

        # if batch_num % 100 == 0:
        #     print(pred)
        #     print(y)

        # Compute loss
        loss = criterion(pred, y.float())
        writer.add_scalar("Loss/train over batches", loss, epoch * n_batches + batch_num)
        total_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        print("[Batch]: {}/{} in {:.5f} seconds. Loss: {}".format(
            batch_num, len(data[0]), time.time() - t, 100**2 * total_loss / (batch_num * len(batch))), end='\r', flush=True)
        t = time.time()


    writer.add_scalar('Loss/train over epochs', total_loss, epoch)
    print()
    print("[Loss]: {:.5f}".format(100**2 * total_loss / (args.batch_size * len(data[0]))))
    return total_loss / (args.batch_size * len(data[0]))


def evaluate(model, data, criterion, device, args, type='Valid'):
    model.eval()
    t = time.time()
    total_loss = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data[0]):
            x = batch
            x_lens = data[1][batch_num]
            y = data[2][batch_num]

            pred, attn = model(x.to(device), x_lens)
            total_loss += float(criterion(pred.float(), y))
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data[0]), time.time() - t), end='\r', flush=True)
            t = time.time()

            if batch_num == 1:
                print(pred)
                print(y)

    print()
    print("[{} loss]: {:.5f}".format(type, 100**2 * total_loss / (len(data[0]) * args.batch_size)))
    return total_loss / (len(data[0]) * args.batch_size)


def main():
    args = make_parser().parse_args()
    print_args(args)

    cuda = torch.cuda.is_available() and args.cuda
    print("Found cuda: ", torch.cuda.is_available())
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    seed_everything(seed=1337, cuda=cuda)

    # init tensorboard writer
    writer = SummaryWriter()


    # Load dataset iterators

    # iters, TEXT, LABEL = dataset_map[args.data](args.batch_size, device=device, vectors=vectors)
    iters = load_data(args.dataset_path, args.embeds_path, args.lang, args.batch_size, device)

    # Some datasets just have the train & test sets, so we just pretend test is valid
    if len(iters) == 4:
        X_str, train_iter, val_iter, test_iter = iters
    else:
        X_str, train_iter, test_iter = iters
        val_iter = test_iter

    print("[Corpus]: train: {}, test: {}".format(
        len(train_iter[0]) * len(train_iter[0][0]), len(test_iter[0])*len(test_iter[0][0])))

    # Define model pipeline
    encoder = Encoder(args.emsize, args.hidden, rnn_type=args.model, nlayers=args.nlayers,
                      dropout=args.drop, bidirectional=args.bi)

    if args.use_pretrained:
        embedding = nn.Linear(args.emsize, args.emsize)
    else:
        with open(args.vocab_path, 'r') as f:
            vocab = json.load(f)
        embedding = nn.Embedding(len(vocab), args.emsize, padding_idx=0)
    attention_dim = args.hidden if not args.bi else 2 * args.hidden
    attention = Attention(attention_dim, attention_dim, attention_dim)
    fc_layer_dims = [attention_dim, 10, 5]
    seq_len = len(train_iter[0][0][0]) # one sentence length
    if args.use_outputs: fc_layer_dims = [seq_len * attention_dim, 500, 250, 50, 10]

    model = Vectorizer(embedding, encoder, fc_layer_dims, attention, concat_out=args.use_outputs, use_attn=args.use_attn)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wdecay, amsgrad=True)

    # Train and validate per epoch
    try:
        best_valid_loss = None

        for epoch in range(1, args.epochs + 1):
            train(model, train_iter, optimizer, criterion, device, args,
                  epoch=epoch-1,
                  writer=writer)
            loss = evaluate(model, val_iter, criterion, device, args, type='Valid')

            # Save model
            torch.save(model, os.path.join(args.save_path, f'model_epoch_{epoch}.pt'))


            if not best_valid_loss or loss < best_valid_loss:
                best_valid_loss = loss

            # log in tensorboard
            writer.add_scalar("Loss/val by epoch", loss, epoch)

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    loss = evaluate(model, test_iter, criterion, device, args, type='Test')
    print("Test loss: ", 100**2 * loss)
    writer.add_scalar("Loss/test final", loss)


    # save Tensorboard logs and close writer
    writer.flush()
    writer.close()

    # Print some sample evaluations
    batch_to_print_X, batch_to_print_X_lens, batch_to_print_Y = val_iter[0][0]
    pred, _ = model(batch_to_print_X, batch_to_print_X_lens)
    print("Predictions: ", pred)
    print("Original: ", batch_to_print_Y)


if __name__ == '__main__':
    main()
