import argparse
import json
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn

from datasets import *
from decoder import *
from encoder import *


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Sequence Generator')
    parser.add_argument('--save_path', type=str, default='saved_models/fr_decoder/model.pt')
    parser.add_argument('--dataset_path', type=str, default='../generate-data/data/train/fr.csv',
                        help='Dataset path')
    parser.add_argument('--lang', type=str, default='fr')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
    parser.add_argument('--embeds_path', type=str, default='../decoder/data/indexed_data.json',
                        help='Embeddings path')
    parser.add_argument('--vocab_path', type=str, default='../decoder/data/vocab.json',
                        help='Embeddings path')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
    parser.add_argument('--hidden', type=int, default=300, # changing hidden to match emsize
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

def train_encoder(fcl, decoder, data, decoder_optimizer, criterion, target_length, args):
    """
    :param model: (nn.Module) model
    :param data: iterator of data
    :param optimizer:
    :param criterion: loss function(pred, actual)
    :param args:
    :return:
    """
    fcl.train()
    decoder.train()
    t = time.time()
    total_loss = 0

    loss = 0
    for batch_num, batch in enumerate(data):
        # x is coordinate, y is output indices
        x = data[1][batch_num].float()  #casting as float so it works with FCL
        y = data[0][batch_num] # reversing x and y here, x is position, y is str

        # Forward pass
        # print(f'initial x : {x}')
        init_hidden = fcl(x).unsqueeze(0)  # hidden dim is (num_layers, batch, hidden_size)
        if args.model == 'LSTM':
            init_hidden = (init_hidden, init_hidden)
        # print(f'x after fcl, hidden batch : {init_hidden}')

        # init decoder input and hidden
        decoder_input = torch.ones(args.batch_size, dtype=torch.long) #init starting tokens, long is the same as ints, which are needed for embedding layer
        decoder_hidden = init_hidden

        print(f'decoder: {decoder}')
        print(f'decoder input size: {decoder_input.size()}')
        print(f'decoder hidden size: {decoder_hidden.size()}')


        # run batch through rnn
        for di in range(1, target_length):
            print(f'rnn loop {di}, before self.decoder')
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) #TODO - handle LSTMs here too
            print(f'rnn loop {di}, after self.decoder')

            # get top index from softmax of previous layer
            topv, topi = decoder_output.topk(1)
            # decoder_input = topi.squeeze().detach()
            decoder_input = topi.detach()
            loss += criterion(decoder_output.float(), y[:,di]) #cast as float so mse comp works
            # if top_i.item() == 2: # if end token
            #     break

        # Compute loss
        # loss = criterion(pred, y) # make sure y is off
        total_loss += loss
        loss.requires_grad = True
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
        decoder_optimizer.step()

        print("[Batch]: {}/{} in {:.5f} seconds. Loss: {}".format(
            batch_num, len(data), time.time() - t, total_loss / (batch_num * len(batch))), end='\r', flush=True)
        t = time.time()

    print()
    print("[Loss]: {:.5f}".format(total_loss / len(data)))
    return total_loss / (args.batch_size * len(data[0]))


def evaluate_encoder(model, data, criterion, type='Valid'):
    model.eval()
    t = time.time()
    total_loss = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data):

            # again, switching x and y here compared to decoder
            x = data[1][batch_num]
            y = batch

            pred = model(x)
            total_loss += float(criterion(pred, y))
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data), time.time() - t), end='\r', flush=True)
            t = time.time()

    print()
    print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
    print(attn)
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
    target_length = train_iter[0][0][0].shape[0] # TODO - double check this is the right length

    # get size of vocab
    vocab = load_json(args.vocab_path)
    output_dims = len(vocab)

    print("[Corpus]: train: {}, test: {}".format(
        len(train_iter[0]) * len(train_iter[0][0]), len(test_iter[0])*len(test_iter[0][0])))


    # Load or define embedding layer
    # if args.use_pretrained:
    #     embedding = nn.Linear(args.emsize, args.emsize)
    # else:
    #     print(f'initialize new embedding: {len(vocab)}')
    #     embedding = nn.Embedding(len(vocab), args.emsize, padding_idx=0)
    #

    # Define model pipeline
    # FCL
    fc_layer_dims = [args.hidden] #output of FC should be h0, first hidden input
    fcl = FC_Encoder(layer_dims=fc_layer_dims)

    # RNN
    decoder = Decoder(output_dims, args.hidden, rnn_type=args.model, nlayers=args.nlayers,
                      dropout=args.drop)


    fcl.to(device)
    decoder.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()

    # fcl_optimizer = torch.optim.Adam(fcl.parameters(), args.lr, amsgrad=True)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), args.lr, amsgrad=True)


    # Train and validate per epoch
    try:
        best_valid_loss = None

        for epoch in range(1, args.epochs + 1):
            print(f'Epoch: {epoch}')
            train_encoder(fcl, decoder, train_iter, decoder_optimizer, criterion, target_length, args)
            loss = evaluate_encoder(model, val_iter, criterion)

            if not best_valid_loss or loss < best_valid_loss:
                best_valid_loss = loss

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    loss = evaluate_encoder(model, test_iter, optimizer, criterion, args, type='Test')
    print("Test loss: ", loss)


if __name__ == '__main__':
    main()








