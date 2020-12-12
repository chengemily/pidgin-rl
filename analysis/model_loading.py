from itertools import chain
import json
import argparse
import os, sys
sys.path.insert(1, "/home/ubuntu/pidgin-rl/model")
print(sys.path)

import time
import numpy as np
import torch
import torch.nn as nn

from datasets import *
from decoder import *
from encoder_v2 import *
from train_encoder_v2 import *



def _load_modules_from_joint(model_path, device):
    '''
    Load components from joint model
    '''
    model_joint = torch.load(model_path,map_location=device)
    embedding, encoder, fc_encoder, decoder, fc = tuple(model_joint.children())
    return embedding, encoder, fc_encoder, decoder, fc


def load_sequence_generator_and_vectorizer(model_path, data, device):
    '''
    Creates Sequence Generator and Vectorizer objects from pre-trained
    components trained using joint training
    
    param: model_path points to saved pytorch model
    return: sequenece generator, vectorizer
    '''
    embedding, encoder, fc_encoder, decoder, fc = _load_modules_from_joint(model_path, device)
    
    # get input params for seq generator and vectorizer
    rnn_encoder, linear_encoder = tuple(encoder.children())
    rnn_type = type(rnn_encoder)
    if isinstance(rnn_encoder, torch.nn.modules.rnn.LSTM): rnn_type = 'LSTM'
    else:rnn_type = 'GRU'
    TARGET_LENGTH = len(data[0][0][0])
    output = VOCAB_SIZE
    rnn_decoder = list(decoder.children())[0]
    
    # create seq_generator object
    seq_gen = seq_generator = Sequence_Generator(embedding, 
                                                 rnn_encoder,
                                                 [rnn_encoder.hidden_size],
                                                  TARGET_LENGTH, 
                                                 output, 
                                                 BATCH_SIZE, 
                                                 VOCAB_SIZE,
                                                  device)
    
    # create vectorizer obbject
    vectorizer = Vectorizer(embedding, rnn_decoder, [rnn_decoder.hidden_size], None)
    
    return seq_gen, vectorizer
    



if __name__=="__main__":
    model_path = "../model/saved_models/en/model_en_pretrained_epoch_10.pt"
    load_sequence_generator_and_vectorizer(model_path)