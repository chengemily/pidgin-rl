from itertools import chain
import argparse
import json
import os, sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.tensorboard import SummaryWriter

from datasets import load_data
from decoder import *
from train_decoder import *
from encoder_v2 import *
from train_encoder_v2 import *
from test_seq_generation import *
from agent import *


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
    parser.add_argument('--model_save_path', type=str, default='saved_models/en/model_joint.pt')
    parser.add_argument('--dataset_path', type=str, default='../generate-data/data_final/train/en.csv')
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--embeds_path', type=str, default='../tokenizer/data_final/indexed_data_words.json',
                        help='Embeddings path')
    parser.add_argument('--vocab_path', type=str, default='../tokenizer/data_final/vocab_words.json',
                        help='Embeddings path')

    # Embedding params
    parser.add_argument('--emsize', type=int, default=20,
                        help='size of word embeddings')
    parser.add_argument('--use_pretrained', action='store_true')

    # Global params
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--cuda', action='store_false',
                        help='[DONT] use CUDA')
    parser.add_argument('--no_tensorboard', action='store_false',
                        help="[DON'T] use tensorboard")

    # Decoder params
    parser.add_argument('--decoder_hidden', type=int, default=50,
                        help='number of hidden units for the RNN encoder')
    parser.add_argument('--decoder_nlayers', type=int, default=1,
                        help='number of layers of the RNN encoder')
    parser.add_argument('--decoder_lr', type=float, default=0.005,
                        help='initial learning rate')
    parser.add_argument('--decoder_wdecay', type=float, default=1e-3,
                        help='weight decay')
    parser.add_argument('--decoder_clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--decoder_drop', type=float, default=0.3,
                        help='dropout')
    parser.add_argument('--decoder_bi', action='store_true',
                        help='[USE] bidirectional encoder')

    parser.add_argument('--use_outputs', action='store_true', help='concat outputs mode')
    parser.add_argument('--use_attn', action='store_true', help='use dot prod attn')

    # Encoder params
    parser.add_argument('--encoder_hidden', type=int, default=50,  # changing hidden to match emsize
                        help='number of hidden units for the RNN decoder')
    parser.add_argument('--encoder_nlayers', type=int, default=1,
                        help='number of layers of the RNN decoder')
    parser.add_argument('--encoder_lr', type=float, default=0.005,
                        help='initial learning rate')
    parser.add_argument('--encoder_clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--encoder_drop', type=float, default=0.3,
                        help='dropout')

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


def train(model, data, optimizer, e_crit, d_crit, device, args, ix_to_word=None, epoch=None):
    model.train()
    t = time.time()
    n_batches = len(data[0])
    decoder_loss, encoder_loss = 0, 0

    for batch_num, batch in enumerate(data[0]):
        # # Define data
        decoder_x = batch # Tensors of indices i.e. sequences
        decoder_x_len = data[1][batch_num] # Lengths of tensors of indices
        decoder_y = data[2][batch_num] # Tensors of outputs vectors
        encoder_x = data[2][batch_num].float()
        encoder_y = batch

        # Forward pass decoder
        dec_pred = model.forward_decoder(decoder_x.to(device), decoder_x_len)

        # Forward pass encoder
        enc_pred = model.forward_encoder(encoder_x.to(device))

        # Compute decoder loss and backprop
        dec_loss = d_crit(dec_pred, decoder_y.float())
        decoder_loss += dec_loss
        optimizer.zero_grad()
        dec_loss.backward()
        optimizer.step() # will update for encoder as well

        # Log decoder loss
        t = time.time()

        with torch.autograd.set_detect_anomaly(True):
            # Compute loss
            enc_loss = e_crit(enc_pred, encoder_y)  # TODO - pred might be the wrong dimensions (switch 1 and 2)
            encoder_loss += enc_loss

            # Backward pass and optimizer step
            optimizer.zero_grad()
            enc_loss.backward()
            optimizer.step()

            # print example sentences
            if ix_to_word and (batch_num % 100 == 0):
                translated_batch = translate_batch(enc_pred, ix_to_word)
                translated_y = translate_batch(encoder_y, ix_to_word)
                print(f'Predicted sentences: \n{translated_batch[:3]}\n')
                print(f'Actual sentence: \n{translated_y[:3]}\n\n')

            # Detach pred?
            enc_pred.detach()

            print("[Batch]: {}/{} in {:.5f} seconds. Encoder Loss: {}. Decoder Loss: {}".format(
                batch_num, len(data[0]), time.time() - t, encoder_loss / (batch_num), 100 ** 2 * decoder_loss / (batch_num * len(batch))), end='\r',
                flush=True)
            t = time.time()

    return decoder_loss / (args.batch_size * len(data[0])), encoder_loss / (args.batch_size * len(data[0]))

def evaluate(model, data, e_crit, d_crit, device, args, type='Valid'):
    model.eval()
    t = time.time()
    encoder_loss, decoder_loss = 0, 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data[0]):
            encoder_x = data[2][batch_num].float()
            encoder_y = batch
            decoder_x = batch
            decoder_x_len = data[1][batch_num]
            decoder_y = data[2][batch_num]

            enc_pred = model.forward_encoder(encoder_x.to(device))
            dec_pred = model.forward_decoder(decoder_x.to(device), decoder_x_len)
            encoder_loss += float(e_crit(enc_pred, encoder_y))
            decoder_loss += float(d_crit(dec_pred, decoder_y))
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data[0]), time.time() - t), end='\r', flush=True)
            t = time.time()

    print()
    return encoder_loss / (len(data[0]) * args.batch_size), decoder_loss / (len(data[0]) * args.batch_size * len(data[0][0]))

def main():
    args = make_parser().parse_args()
    print_args(args)

    # get ix_to_word map
    IX_TO_WORD = create_ix_to_vocab_map(args.vocab_path)

    # Cuda, seeding-----------------------------------------------------------------------------------------------------
    cuda = torch.cuda.is_available() and args.cuda
    print("Found cuda: ", torch.cuda.is_available())
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    seed_everything(seed=1337, cuda=cuda)

    # Load dataset iterators--------------------------------------------------------------------------------------------
    iters = load_data(args.dataset_path, args.embeds_path, args.lang, args.batch_size, device)

    # Some datasets just have the train & test sets, so we just pretend test is valid
    if len(iters) == 4:
        X_str, train_iter, val_iter, test_iter = iters
    else:
        X_str, train_iter, test_iter = iters
        val_iter = test_iter

    print("[Corpus]: train: {}, test: {}".format(
        len(train_iter[0]) * len(train_iter[0][0]), len(test_iter[0])*len(test_iter[0][0])))

    # get size of vocab
    vocab = load_json(args.vocab_path)
    output_dims = len(vocab)

    # Load embedding, to be shared -------------------------------------------------------------------------------------
    if args.use_pretrained:
        embedding = nn.Linear(args.emsize, args.emsize)
    else:
        with open(args.vocab_path, 'r') as f:
            vocab = json.load(f)
        embedding = nn.Embedding(len(vocab), args.emsize, padding_idx=0)

    # Define decoder model pipeline ------------------------------------------------------------------------------------
    # Decoder RNN module
    decoder_rnn = Encoder(args.emsize, args.decoder_hidden, rnn_type=args.model, nlayers=args.decoder_nlayers,
                      dropout=args.decoder_drop, bidirectional=args.decoder_bi)

    # Attention
    attention_dim = args.decoder_hidden if not args.decoder_bi else 2 * args.decoder_hidden

    # Fully connected network using either last hidden state or outputs
    dec_fc_layer_dims = [attention_dim, 10, 5]
    seq_len = len(train_iter[0][0][0]) # one sentence length
    if args.use_outputs: dec_fc_layer_dims = [seq_len * attention_dim, 500, 250, 50, 10]
    fc_decoder = FC(dec_fc_layer_dims)

    # Define loss and optimizer
    dec_criterion = nn.MSELoss()

    # Define encoder model and pipeline---------------------------------------------------------------------------------
    # FCL
    enc_fc_layer_dims = [args.encoder_hidden]  # output of FC should be h0, first hidden input
    fc_encoder = FC_Encoder(enc_fc_layer_dims)

    # RNN
    encoder_rnn = Decoder(output_dims, args.encoder_hidden, args.emsize, rnn_type=args.model, nlayers=args.encoder_nlayers,
                      dropout=args.encoder_drop)

    # Sequence Generator
    target_length = len(train_iter[0][0][0])

    # Define loss and optimizer
    enc_criterion = nn.CrossEntropyLoss(ignore_index=0)

    model = Agent(embedding, encoder_rnn, fc_encoder, decoder_rnn, fc_decoder, args.batch_size, output_dims, target_length, device)
    optimizer = torch.optim.Adam(model.parameters(), args.encoder_lr, amsgrad=True)
    model.to(device)

    # Training and validation loop--------------------------------------------------------------------------------------
    try:
        best_valid_dec_loss = None
        best_valid_enc_loss = None

        for epoch in range(1, args.epochs + 1):
            # train(encoder, decoder, train_iter, enc_optimizer, dec_optimizer, enc_criterion, dec_criterion, device, args, ix_to_word=IX_TO_WORD, epoch=epoch)
            train(model, train_iter, optimizer, enc_criterion, dec_criterion, device, args, ix_to_word=IX_TO_WORD, epoch=epoch)
            dec_loss, enc_loss = evaluate(model, val_iter, enc_criterion, dec_criterion, device, args, type='Valid')

            if not best_valid_dec_loss or dec_loss < best_valid_dec_loss:
                best_valid_dec_loss = dec_loss
            if not best_valid_enc_loss or enc_loss < best_valid_enc_loss:
                best_valid_enc_loss = enc_loss

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    dec_test_loss, enc_test_loss = evaluate(model, test_iter, enc_criterion, dec_criterion, device, args, type='Test')
    print("best decoder val loss: ", 100**2 * best_valid_dec_loss)
    print("best encoder val loss: ", best_valid_enc_loss)
    print("decoder test loss: ", 100**2 * dec_test_loss)
    print("encoder test loss: ", enc_test_loss)

    # Save model
    torch.save(model, args.model_save_path)

if __name__ == '__main__':
    main()