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


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
    parser.add_argument('--decoder_save_path', type=str, default='saved_models/en_decoder/model_joint.pt')
    parser.add_argument('--encoder_save_path', type=str, default='saved_models/en_encoder/model_joint.pt')
    parser.add_argument('--dataset_path', type=str, default='../generate-data/data/train/en.csv')
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--embeds_path', type=str, default='../tokenizer/data/indexed_data.json',
                        help='Embeddings path')
    parser.add_argument('--vocab_path', type=str, default='../tokenizer/data/vocab.json',
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
    parser.add_argument('--decoder_hidden', type=int, default=20,
                        help='number of hidden units for the RNN encoder')
    parser.add_argument('--decoder_nlayers', type=int, default=1,
                        help='number of layers of the RNN encoder')
    parser.add_argument('--decoder_lr', type=float, default=1e-3,
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
    parser.add_argument('--encoder_hidden', type=int, default=20,  # changing hidden to match emsize
                        help='number of hidden units for the RNN decoder')
    parser.add_argument('--encoder_nlayers', type=int, default=1,
                        help='number of layers of the RNN decoder')
    parser.add_argument('--encoder_lr', type=float, default=1e-3,
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


def train(encoder, decoder, data, e_optimizer, d_optimizer, e_criterion, d_criterion, device, args, ix_to_word=None, epoch=1, writer=None):
    """
    TODO: make this joint training
    :param decoder: (nn.Module) decoder taking a string to a vector
    :param encoder: (nn.Module) encoder taking a vec to a string
    :param data: tuple of iterators of data (dataX, dataY)
    :param optimizer:
    :param criterion: loss function(pred, actual)
    :param args:
    :return:
    """
    # decoder.train()
    encoder.train()
    t = time.time()
    n_batches = len(data[0])
    decoder_loss, encoder_loss = 0, 0
    attn_maps = []

    for batch_num, batch in enumerate(data[0]):
        # decoder.zero_grad()
        encoder.zero_grad()

        # # Define data
        # decoder_x = batch # Tensors of indices i.e. sequences
        # decoder_x_len = data[1][batch_num] # Lengths of tensors of indices
        # decoder_y = data[2][batch_num] # Tensors of outputs vectors
        encoder_x = data[2][batch_num].float()
        encoder_y = batch
        #
        # # Forward pass decoder
        # dec_pred, _ = decoder(decoder_x.to(device), decoder_x_len)
        #
        # # Compute decoder loss and backprop
        # dec_loss = d_criterion(dec_pred, decoder_y.float())
        # decoder_loss += dec_loss
        # d_optimizer.zero_grad()
        # dec_loss.backward()
        # torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.decoder_clip)
        # d_optimizer.step() # will update for encoder as well
        #
        # # Log decoder loss
        # print("[Batch]: {}/{} in {:.5f} seconds. Decoder Loss: {}".format(
        #     batch_num, len(data[0]), time.time() - t, 100 ** 2 * decoder_loss / (batch_num * len(batch))), end='\r',
        #     flush=True)
        # print()
        # t = time.time()

        with torch.autograd.set_detect_anomaly(True):
            # Encoder fwd pass
            enc_pred = encoder(encoder_x.to(device))

            # Compute loss
            enc_loss = e_criterion(enc_pred, encoder_y)  # TODO - pred might be the wrong dimensions (switch 1 and 2)
            if writer:
                n_batches = len(data[0])
                writer.add_scalar("Loss/train", enc_loss, epoch * n_batches + batch_num)
            encoder_loss += enc_loss

            # Backward pass and optimizer step
            encoder_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.encoder_clip)
            e_optimizer.step()

            # print example sentences
            if ix_to_word and (batch_num % 100 == 0):
                translated_batch = translate_batch(enc_pred, ix_to_word)
                translated_y = translate_batch(encoder_y, ix_to_word)
                print(f'Predicted sentences: \n{translated_batch[:3]}\n')
                print(f'Actual sentence: \n{translated_y[:3]}\n\n')

            # Detach pred?
            enc_pred.detach()
            print("[Batch]: {}/{} in {:.5f} seconds. Encoder Loss: {}".format(
                batch_num, len(data[0]), time.time() - t, encoder_loss / (batch_num * len(batch))), end='\r', flush=True)
            print()
            t = time.time()

    print()
    # print("[Decoder Loss]: {:.5f}".format(100**2 * decoder_loss / (args.batch_size * len(data[0]))))
    print("[Encoder Loss]: {:.5f}".format(100 ** 2 * encoder_loss / (args.batch_size * len(data[0]))))
    return decoder_loss / (args.batch_size * len(data[0])), encoder_loss / (args.batch_size * len(data[0]))


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
    attention = Attention(attention_dim, attention_dim, attention_dim)

    # Fully connected network using either last hidden state or outputs
    dec_fc_layer_dims = [attention_dim, 10, 5]
    seq_len = len(train_iter[0][0][0]) # one sentence length
    if args.use_outputs: dec_fc_layer_dims = [seq_len * attention_dim, 500, 250, 50, 10]

    # Complete pipeline and put on GPU
    decoder = Vectorizer(embedding, decoder_rnn, dec_fc_layer_dims, attention, concat_out=args.use_outputs, use_attn=args.use_attn)
    decoder.to(device)

    # Define loss and optimizer
    dec_criterion = nn.MSELoss()
    dec_optimizer = torch.optim.Adam(decoder.parameters(), args.decoder_lr, weight_decay=args.decoder_wdecay, amsgrad=True)

    # Define encoder model and pipeline---------------------------------------------------------------------------------
    # FCL
    enc_fc_layer_dims = [args.encoder_hidden]  # output of FC should be h0, first hidden input

    # RNN
    encoder_rnn = Decoder(output_dims, args.encoder_hidden, embedding, rnn_type=args.model, nlayers=args.encoder_nlayers,
                      dropout=args.encoder_drop)

    # Sequence Generator
    target_length = len(train_iter[0][0][0])
    encoder = Sequence_Generator(embedding,
                                      encoder_rnn,
                                      enc_fc_layer_dims,
                                      target_length,
                                      output_dims,
                                      args.batch_size,
                                      output_dims,
                                      rnn_type=args.model,
                                      device=device)

    # put all models on GPU
    encoder.to(device)

    # Define loss and optimizer
    enc_criterion = nn.NLLLoss(ignore_index=0)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), args.encoder_lr, amsgrad=True)


    # Training and validation loop--------------------------------------------------------------------------------------
    try:
        best_valid_dec_loss = None
        best_valid_enc_loss = None

        for epoch in range(1, args.epochs + 1):
            train(encoder, decoder, train_iter, enc_optimizer, dec_optimizer, enc_criterion, dec_criterion, device, args, ix_to_word=IX_TO_WORD, epoch=epoch)
            dec_loss = evaluate(decoder, val_iter, dec_criterion, device, args, type='Valid')
            enc_loss = evaluate_encoder(encoder, val_iter, enc_criterion, device, args, type='Valid')

            if not best_valid_dec_loss or dec_loss < best_valid_dec_loss:
                best_valid_dec_loss = dec_loss
            if not best_valid_enc_loss or enc_loss < best_valid_enc_loss:
                best_valid_enc_loss = enc_loss

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    dec_test_loss = evaluate(decoder, test_iter, dec_criterion, device, args, type='Test')
    enc_test_loss = evaluate(encoder, test_iter, enc_criterion, device, args, type='Test')
    print("best decoder val loss: ", 100**2 * best_valid_dec_loss)
    print("best encoder val loss: ", best_valid_enc_loss)
    print("decoder test loss: ", 100**2 * dec_test_loss)
    print("encoder test loss: ", enc_test_loss)

    # Save model
    torch.save(encoder, args.encoder_save_path)
    torch.save(decoder, args.decoder_save_path)

if __name__ == '__main__':
    main()