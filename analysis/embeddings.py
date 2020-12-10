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
import matplotlib.pyplot as plt

from datasets import *
from decoder import *
from encoder_v2 import *
from train_encoder_v2 import *


# tSNE embeddings

# PCA embeddings

# Plot embeddings + labels

# Cos-sim matrix of embeddings

# Center embeddings
def center_embeds(embedding_of_all_V, monolingual_indices):
    """
    Centers embeddings with respect
    :param embedding_of_all_V:
    :return:
    """

# Can we learn word alignments in an unsupervised way?

if __name__=="__main__":
    pass