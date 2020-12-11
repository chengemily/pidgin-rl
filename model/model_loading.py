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
from agent import *


def load_modules_from_joint(model_path, device):
    '''
    Load components from joint model
    '''
    model_joint = torch.load(model_path,map_location=device)
    embedding, encoder, fc_encoder, decoder, fc = tuple(model_joint.children())
    return embedding, encoder, fc_encoder, decoder, fc


def load_sequence_generator_and_vectorizer(model_path, device, target_length, batch_size, end2end=False):
    '''
    Creates Sequence Generator and Vectorizer objects from pre-trained
    components trained using joint training
    
    param: model_path points to saved pytorch model
    :param end2end whether we're doing it in pidgin mode (needs modification)
    return: sequenece generator, vectorizer
    '''
    embedding, encoder, fc_encoder, decoder, fc_decoder = load_modules_from_joint(model_path, device)

    # get input params for seq generator and vectorizer
    rnn_encoder, linear_encoder = tuple(encoder.children())
    rnn_type = type(rnn_encoder)
    if isinstance(rnn_encoder, torch.nn.modules.rnn.LSTM): rnn_type = 'LSTM'
    else:rnn_type = 'GRU'
    vocab_size = int(embedding.weight.size()[0]) # vocab size
    rnn_decoder = list(decoder.children())[0]
    
    # create seq_generator object
    # end2end=True will make the outputs have a gumbel-softmax activation, allowing end2end differentiation w a decoder.
    seq_gen = seq_generator = Sequence_Generator(embedding, 
                                                 rnn_encoder,
                                                 fc_encoder,
                                                  target_length,
                                                 vocab_size,
                                                 batch_size,
                                                 vocab_size,
                                                  device, end2end=end2end)
    
    # create vectorizer obbject
    vectorizer = Vectorizer(embedding, rnn_decoder, fc_decoder, None, end2end=end2end)
    
    return seq_gen, vectorizer
    
def load_agent_convert_end2end(model_path, device, target_length, batch_size,):
    """
    Loads a non-straight through differentiable model and converts it to be differentiable end to end
    :param model_path:
    :param device:
    :param target_length:
    :param batch_size:
    :param vocab_size:
    :return:
    """
    embed, encoder, fc_encoder, decoder, fc_decoder = load_modules_from_joint(model_path, device)
    vocab_size = int(embed.weight.size()[0])
    return Agent(
        embed, encoder, fc_encoder, decoder, fc_decoder, batch_size, vocab_size, target_length, device, end2end=True)


if __name__=="__main__":
    model_path = "../model/saved_models/en/model_en_pretrained_epoch_10.pt"
    load_sequence_generator_and_vectorizer(model_path)