from itertools import chain
import json
import argparse
import os, sys

import time
import numpy as np
import torch
import torch.nn as nn

from datasets import *
from decoder import *
from encoder_v2 import *
from train_encoder_v2 import *
from model_loading import *
from torch.autograd import Variable


from agent import *


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
    parser.add_argument('--dataset_path_1', type=str, default='../generate-data/data_final/train/fr.csv')
    parser.add_argument('--dataset_path_2', type=str, default='../generate-data/data_final/train/en.csv')
    parser.add_argument('--lang_1', type=str, default='fr')
    parser.add_argument('--lang_2', type=str, default='en')
    parser.add_argument('--tensorboard_suffix_1', type=str, default='model_from_fr')
    parser.add_argument('--tensorboard_suffix_2', type=str, default='model_from_en')
    parser.add_argument('--embeds_path', type=str, default='../tokenizer/data_final/indexed_data_words.json',
                        help='Embeddings path')
    parser.add_argument('--vocab_path', type=str, default='../tokenizer/data_final/vocab_words.json',
                        help='Embeddings path')
    parser.add_argument('--no_tensorboard', action='store_false',
                        help="[DON'T] use tensorboard")
    parser.add_argument('--cuda', action='store_false',
                        help='[DONT] use CUDA')

    # Loading pretrained model
    parser.add_argument('--load_path_1', type=str, default="saved_models/fr/model_fr_pretrained_epoch_15.pt")
    parser.add_argument('--load_path_2', type=str, default="saved_models/en/model_en_pretrained_epoch_15.pt")
    parser.add_argument('--save_path', type=str, default="saved_models/end2end/")

    # Embedding params
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--emsize', type=int, default=20,
                        help='size of word embeddings')
    parser.add_argument('--clip', type=float, default=5)

    # Global params
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
    parser.add_argument('--epochs', type=int, default=15,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--target_length', type=int, default=24, help='max length of sequence (seen to be 24)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='initial learning rate')

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


def get_lengths(seq, device):
    """
    Get lengths of batch of sequences
    :param seq: batch x V x seq_len
    :return:
    """
    # print(seq.size())
    eos = torch.zeros(seq.size()[1], dtype=torch.float, device=torch.device('cuda:0'))
    lengths = []
    for sentence in seq:
        for i, word in enumerate(torch.transpose(sentence, 0, 1)):
            if torch.argmax(word) == 2 or i == seq.size()[-1] - 1:
                lengths.append(i + 1)
                # print(lengths)
                # input()
                break
    # print(len(lengths))
    # print("LENGTHS", lengths)
    return torch.tensor(lengths, device=device)


def print_sentences(pred, inp, model, ix_to_word):
    # print example sentences
    # input(pred.size())
    # pred = torch.argmax(pred, dim=1)
    translated_batch = translate_batch(pred, ix_to_word)
    print(f'Original instructions: \n{inp[:3]}')
    print(f'Predicted sentences: \n{translated_batch[:3]}\n')


def train(model_1, model_2, data_1, data_2, model_1_optimizer, model_2_optimizer, crit, device, args, ix_to_word, epoch=None, writer=None):
    """

    :param model_1:
    :param model_2:
    :param data_1:
    :param data_2:
    :param optimizer:
    :param crit:
    :param device:
    :param args:
    :param ix_to_word:
    :param epoch:
    :param writer:
    :return:
    """
    model_1.train()
    model_2.train()
    t = time.time()
    total_loss_1, total_loss_2 = 0, 0
    n_batches = min(len(data_1[0]), len(data_2[0]))
    # embed_1_weight = model_1.embedding.weight.clone()

    for batch_num in range(n_batches):
        # Original vectors
        model_1_x = data_1[2][batch_num].float()
        model_1_y = data_1[2][batch_num].float()
        model_2_x = data_2[2][batch_num].float()  # get [x,y] vectors for both models
        model_2_y = data_2[2][batch_num].float()

        # Produce strings
        model_1_str = model_1.forward_encoder(model_1_x.to(device))
        model_2_str = model_2.forward_encoder(model_2_x.to(device))

        # convert to indices
        seq_len_1 = get_lengths(model_1_str, device)
        seq_len_2 = get_lengths(model_2_str, device)

        # Decode into vectors
        # input(model_1_str.size())
        model_1_pred = model_2.forward_decoder(model_1_str, seq_len_1)
        model_2_pred = model_1.forward_decoder(model_2_str, seq_len_2)

        # Compute loss
        loss_1 = crit(model_1_pred, model_1_y)
        loss_2 = crit(model_2_pred, model_2_y)
        total_loss_1 += loss_1
        total_loss_2 += loss_2
        writer[0].add_scalar("Original {} Model Loss/train over batches".format(args.lang_1), 100 ** 2 * loss_1 / args.batch_size,
                          epoch * n_batches + batch_num)
        writer[1].add_scalar("Original {} Model Loss/train over batches".format(args.lang_2), 100 ** 2 * loss_2 / args.batch_size,
                          epoch * n_batches + batch_num)

        # Backprop
        model_1_optimizer.zero_grad()
        loss_1.backward()
        torch.nn.utils.clip_grad_norm_(model_1.parameters(), args.clip)
        loss_2.backward()
        torch.nn.utils.clip_grad_norm_(model_2.parameters(), args.clip)
        model_1_optimizer.step()

        # new_embed_w = model_1.embedding.weight
        # for name, param in list(model_1.named_parameters()):
        #     print("Param: ", name)
        #     print("PAram grad is not none: ", param.grad is not None)
        # input()
        # print([param.grad is not None for param in list(model_1.parameters())])
        # print([param.grad is not None for param in list(model_2.parameters())])
        # print([param.requires_grad for param in list(model_1.parameters())])
        # print([param.requires_grad for param in list(model_2.parameters())])
        # assert not torch.equal(new_embed_w, embed_1_weight)
        # embed_1_weight = new_embed_w

        # print example sentences
        if batch_num % 1000 == 0:
            print("SENTENCES FOR {}".format(args.lang_1))
            print_sentences(model_1_str, model_1_x, model_1, ix_to_word)
            print("PREDIECTIONS FOR {}".format(args.lang_1))
            print(model_1_pred[:3])

            print("\nSENTENCES FOR {}".format(args.lang_2))
            print_sentences(model_2_str, model_2_x, model_2, ix_to_word)
            print("PREDIECTIONS FOR {}".format(args.lang_2))
            print(model_2_pred[:3])


        print("[Batch]: {}/{} in {:.5f} seconds. Starting {} Loss: {}. Starting {} Loss: {}".format(
            batch_num, len(data_1[0]), time.time() - t,
            args.lang_2,
            100 ** 2 * loss_2 / (args.batch_size),
            args.lang_1,
            100 ** 2 * loss_1 / (args.batch_size)),
            end='\r',
            flush=True)
        t = time.time()

    writer[0].add_scalar('Model 1 loss/train over epochs', total_loss_1, epoch)
    writer[1].add_scalar('Model 2 loss/train over epochs', total_loss_2, epoch)

    return 100**2 * total_loss_1 / (args.batch_size * n_batches), 100**2 * total_loss_2 / (args.batch_size *n_batches)


def evaluate(model_1, model_2, data_1, data_2, crit, device, args, type='Valid'):
    """

    :param model_1:
    :param model_2:
    :param data_1: the data for testing on model 1
    :param data_2: the data for testing on model 2
    :param crit:
    :param device:
    :param args:
    :param type:
    :return:
    """
    model_1.eval()
    model_2.eval()
    t = time.time()
    n_batches = min(len(data_1[0]), len(data_2[0]))
    loss_1, loss_2 = 0, 0
    with torch.no_grad():
        for batch_num in range(n_batches):
            # Original vectors
            model_1_x = data_1[2][batch_num].float()
            model_1_y = data_1[2][batch_num].float()
            model_2_x = data_2[2][batch_num].float() # get [x,y] vectors for both models
            model_2_y = data_2[2][batch_num].float()

            # Produce strings
            model_1_str = model_1.forward_encoder(model_1_x.to(device))
            model_2_str = model_1.forward_encoder(model_2_x.to(device))

            # convert to indices
            seq_len_1 = get_lengths(model_1_str, device)
            seq_len_2 = get_lengths(model_2_str, device)

            # Decode into vectors
            model_1_pred = model_2.forward_decoder(model_1_str, seq_len_1)
            model_2_pred = model_1.forward_decoder(model_2_str, seq_len_2)

            loss_1 += float(crit(model_1_pred, model_1_y))
            loss_2 += float(crit(model_2_pred, model_2_y))
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data_1[0]), time.time() - t), end='\r', flush=True)
            t = time.time()

    print()
    return 100**2 * loss_1 / (args.batch_size * n_batches), 100**2 * loss_2 / (args.batch_size * n_batches)


def main():
    args = make_parser().parse_args()
    print_args(args)

    # CUDA, seeding
    cuda = torch.cuda.is_available() and args.cuda
    print("Found cuda: ", torch.cuda.is_available())
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    seed_everything(seed=1337, cuda=cuda)

    # init tensorboard writer
    writer_1 = SummaryWriter(comment=args.tensorboard_suffix_1)
    writer_2 = SummaryWriter(comment=args.tensorboard_suffix_2)

    # get ix_to_word map
    IX_TO_WORD = create_ix_to_vocab_map(args.vocab_path)

    # Load dataset iterators--------------------------------------------------------------------------------------------
    iters_1 = load_data(args.dataset_path_1, args.embeds_path, args.lang_1, args.batch_size, device)
    iters_2 = load_data(args.dataset_path_2, args.embeds_path, args.lang_1, args.batch_size, device)

    # Some datasets just have the train & test sets, so we just pretend test is valid
    X_str_1, train_iter_1, test_iter_1 = iters_1
    X_str_2, train_iter_2, test_iter_2 = iters_2
    val_iter_1, val_iter_2 = test_iter_1, test_iter_2

    print("[{} Corpus]: train: {}, test: {}".format(args.lang_1,
        len(train_iter_1[0]) * len(train_iter_1[0][0]), len(test_iter_1[0]) * len(test_iter_1[0][0])))
    print("[{} Corpus]: train: {}, test: {}".format(args.lang_2,
        len(train_iter_2[0]) * len(train_iter_2[0][0]), len(test_iter_2[0]) * len(test_iter_2[0][0])))

    # Load in new models (will see that Agent code has changed-- this is intentional)
    model_1 = load_agent_convert_end2end(args.load_path_1, device, args.target_length, args.batch_size)
    model_2 = load_agent_convert_end2end(args.load_path_2, device, args.target_length, args.batch_size)
    model_1.to(device)
    model_2.to(device)

    # Define criterion, optimizer
    criterion = nn.MSELoss() # We will be using the same MSE throughout.
    model_1_optimizer = torch.optim.Adam(chain(model_1.parameters(), model_2.parameters()), args.lr, amsgrad=True)
    model_2_optimizer = torch.optim.Adam(model_2.parameters(), args.lr, amsgrad=True)

    # Training and validation loop--------------------------------------------------------------------------------------
    try:
        best_loss_1 = np.inf
        best_loss_2 = np.inf

        for epoch in range(1, args.epochs + 1):
            train(model_1, model_2, train_iter_1, train_iter_2, model_1_optimizer, model_2_optimizer, criterion, device, args, IX_TO_WORD,
                  epoch=epoch, writer=[writer_1, writer_2])
            loss_1, loss_2 = evaluate(model_1, model_2, val_iter_1, val_iter_2, criterion, device, args, type='Valid')

            writer_1.add_scalar("{} Agent to {} Loss/val by epoch".format(args.lang_1, args.lang_2), loss_1, epoch)
            writer_2.add_scalar("{} Agent to {} Loss/val by epoch".format(args.lang_2, args.lang_1), loss_2, epoch)

            # save model
            torch.save(model_1, os.path.join(args.save_path, f'{args.tensorboard_suffix_1}_epoch_{epoch}.pt'))
            torch.save(model_2, os.path.join(args.save_path, f'{args.tensorboard_suffix_2}_epoch_{epoch}.pt'))

            if loss_1 < best_loss_1:
                best_loss_1 = loss_1
            if loss_2 < best_loss_2:
                best_loss_2 = loss_2

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    test_loss_1, test_loss_2 = evaluate(model_1, model_2, val_iter_1, val_iter_2, criterion, device, args, type='Valid')
    print("best starting from {} val loss: ".format(args.lang_1), best_loss_1)
    print("best starting from {} val loss: ".format(args.lang_2), best_loss_2)
    print("starting from {} test loss: ".format(args.lang_1), test_loss_1)
    print("starting from {} test loss: ".format(args.lang_2), test_loss_2)


if __name__ == '__main__':
    main()

