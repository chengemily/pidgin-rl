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

def train_encoder(fcl, decoder, data, decoder_optimizer, criterion, target_length, device, args):
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

    loss = 0 # is this redundant with total_loss defined? -> this one will store criterion, total_loss stores cumulative loss

    for batch_num, batch in enumerate(data[0]):
        # x is coordinate, y is output indices
        # print(f'batch size: {len(batch)}')

        x = data[2][batch_num].float().to(device) # make the coordinates the predictors x
        y = batch.to(device) # label (indices) is the word embeddings

        print(f'initial y:{y}')
        print(f'initial y shape : {y.size()}')
        # Forward pass
        print(f'initial x shape : {x.size()}')
        print(f'initial x  : {x}')
        with torch.autograd.set_detect_anomaly(True):
            init_hidden = fcl(x).unsqueeze(0).to(device)  # hidden dim is (num_layers, batch, hidden_size)

            if args.model == 'LSTM':
                init_hidden = (init_hidden, init_hidden)
            # print(f'x after fcl, hidden batch : {init_hidden}')

            # init decoder input and hidden
            decoder_input = torch.ones(args.batch_size, 1, dtype=torch.long).to(device) #init starting tokens, long is the same as ints, which are needed for embedding layer
            decoder_hidden = init_hidden

            # if isinstance(decoder_hidden, tuple):
            #     print(f'decoder hidde: size: {decoder_hidden[0].size()}')
            # else: print(f'decoder hidden size: {decoder_hidden.size()}')


            # run batch through rnn
            for di in range(1, target_length-1): # start with 1 to predict first non-cls word
                # print(f'rnn loop {di}, before self.decoder')
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) #TODO - handle LSTMs here too
                # print(f'rnn loop {di}, after self.decoder')
                # 
                # print(f'decoder output: {decoder_output}')
                # print(f'decoder output size: {decoder_output.size()}')
                # print(f'y: {y[:, di]}')
                # print(f'y size: {y[:, di].size()}')

                # take NLL loss
                pred = decoder_output.float().squeeze()
                target = y[:,di]
                loss += criterion(pred, target) # Make pred [batch, embed] and target [batch,]

                # get top index from softmax of previous layer
                topv, topi = decoder_output.topk(1) # taking argmax
                decoder_input = topi.view(-1,1).detach() # remove unneeded dimension


            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
            decoder_optimizer.step()

            print("[Batch]: {}/{} in {:.5f} seconds. Loss: {}".format(
                batch_num, len(data), time.time() - t, loss / (batch_num * len(batch))), end='\r', flush=True)
            t = time.time()

    print()
    print("[Loss]: {:.5f}".format(loss / len(data)))
    return loss / (args.batch_size * len(data[0]))


def evaluate_encoder(fcl, decoder, data, criterion, target_length, args, type='Valid'):
    model.eval()
    t = time.time()
    total_loss = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data[0]):

            x = data[2][batch_num].float().to(device)
            y = batch.to(device)

            init_hidden = fcl(x).unsqueeze(0).to(device)  # hidden dim is (num_layers, batch, hidden_size)

            if args.model == 'LSTM':
                init_hidden = (init_hidden, init_hidden)

            # init decoder input and hidden
            decoder_input = torch.ones(args.batch_size, 1, dtype=torch.long).to(device) #init starting tokens, long is the same as ints, which are needed for embedding layer
            decoder_hidden = init_hidden

            # run batch through rnn
            for di in range(1, target_length-1): # start with 1 to predict first non-cls word
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) #TODO - handle LSTMs here too

                # take NLL loss
                pred = decoder_output.float().squeeze()
                target = y[:,di]
                total_loss += criterion(pred, target) # Make pred [batch, embed] and target [batch,]

                # get top index from softmax of previous layer
                topv, topi = decoder_output.topk(1) # taking argmax
                decoder_input = topi.view(-1,1).detach() # remove unneeded dimension


            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data), time.time() - t), end='\r', flush=True)
            t = time.time()


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
    target_length = len(train_iter[0][0][0]) # TODO - double check this is the right length
    print(f'target length: {target_length}')

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
    embedding = nn.Embedding(args.emsize, args.hidden)  # 1st param - size of vocab, 2nd param - size of embedding vector
    embedding.to(device) # TODO - double check that I need to do this

    # Define model pipeline
    # FCL
    fc_layer_dims = [args.hidden] #output of FC should be h0, first hidden input
    fcl = FC_Encoder(layer_dims=fc_layer_dims)

    # RNN
    decoder = Decoder(output_dims, args.hidden, embedding, rnn_type=args.model, nlayers=args.nlayers,
                      dropout=args.drop) # TODO - more thoroughly check this

    # put all models on GPU
    fcl.to(device)
    decoder.to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss()

    # fcl_optimizer = torch.optim.Adam(fcl.parameters(), args.lr, amsgrad=True)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), args.lr, amsgrad=True)


    # Train and validate per epoch
    try:
        best_valid_loss = None

        for epoch in range(1, args.epochs + 1):
            print(f'Epoch: {epoch}')
            train_encoder(fcl, decoder, train_iter, decoder_optimizer, criterion, target_length, device, args)
            loss = evaluate_encoder(model, val_iter, criterion)

            if not best_valid_loss or loss < best_valid_loss:
                best_valid_loss = loss

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    loss = evaluate_encoder(model, test_iter, optimizer, criterion, args, type='Test')
    print("Test loss: ", loss)


if __name__ == '__main__':
    main()








