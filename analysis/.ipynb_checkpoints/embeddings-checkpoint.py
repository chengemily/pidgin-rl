from itertools import chain
import json
import argparse
import os, sys
# sys.path.insert(1, "/home/ubuntu/pidgin-rl/model")
sys.path.insert(1, '../model')
# print(sys.path)

import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

import pickle
import signal
import argparse
import traceback
import json
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix

import pandas as pd



from datasets import *
from decoder import *
from encoder_v2 import *
from train_encoder_v2 import *


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Sequence Generator')
    parser.add_argument('--lang', type=str, default='en',
                       help='choose from [en, fr]')
    parser.add_argument('--model_type', type=str, default='encoder',
                        help='choose from [encoder, decoder]')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='which model checkpoint to analyze, default is -1 (the last)')
    parser.add_argument('--last_epoch', type=int, default=9,
                        help='last epoch to analyze')
    parser.add_argument('--embeds_path', type=str, default='../tokenizer/data_final/indexed_data_words.json',
                        help='Embeddings path')
    parser.add_argument('--vocab_path', type=str, default="../tokenizer/data_final/vocab_words.json",
                        help='Embeddings path')
    parser.add_argument('--cuda', action='store_false',
                        help='[DONT] use CUDA')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')

    return parser




# tSNE embeddings

def tSNE(df):
    """
    t-SNEs df into 2 dimensions for visualization
    """
    X_embed = TSNE(n_components=2).fit_transform(df)
    print('t-SNEd into shape:', X_embed.shape)

    return X_embed


# PCA embeddings

def PCA_(n, df):
    """
    PCAs df into n-dimensional df. Centers data automatically
    """
    pca = PCA(n_components=n)
    pca_df = pd.DataFrame(pca.fit_transform(np.array(df)))
    print('PCAed into shape: ', pca_df.shape)
    return pca_df


# Plot embeddings + labels


def plot_embeds(embeds, names, title='tSNE Visualization of Embeddings'):
    """
    Plots embeddings with their corresponding names.

    embeds: N x 2 df where N[i] is a point to plot and names[i] is the corresponding label

    """
    embeds = np.array(embeds)
    for i, embed in enumerate(embeds):
        plt.scatter(embed[0], embed[1])
        plt.text(embed[0] + 0.05, embed[1] - 0.07, names[i], fontsize=9)

    plt.title(title)
    plt.show()

# Cos-sim matrix of embeddings
# Distance matrix

def plot_matrix(mat, classes, title):
    fig, ax = plt.subplots()
    im = ax.imshow(mat)
    plt.title(title)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(classes, {'fontsize': 7})
    ax.set_yticklabels(classes, {'fontsize': 7})

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.colorbar(im)
    plt.show()

def vis_distance_matrix(df, classes, title, cos=True):
    """
    Visualize pairwise cosine distances between rows of the df.
    df should be a pandas dataframe of embedding vectors.
    """
    embeds = np.array(df)

    if cos:
        embeds = normalize(embeds, norm='l2', axis=1, copy=True, return_norm=False)
    dists = distance_matrix(embeds, embeds, p=2)
    
    plot_matrix(dists, classes, title)

    return dists


# Center embeddings
def center_embeds(embedding_of_all_V, monolingual_indices):
    """
    Centers embeddings with respect
    :param embedding_of_all_V:
    :return:
    """

# Can we learn word alignments in an unsupervised way?



                        

def main():
    args = make_parser().parse_args()

    # init parameters
    LANG = args.lang
    BATCH_SIZE = args.batch_size
    DATASET_PATH = '../generate-data/data_final/train/{}.csv'.format(LANG)
    INDEXED_DATA_PATH = args.embeds_path # dataset indexed
    VOCAB_PATH = args.vocab_path
    N = args.last_epoch # last epoch we want
    model_type = args.model_type  # Decoder or encoder
    checkpoint = args.checkpoint


    # Load Data
    dataset = pd.read_csv(DATASET_PATH).drop(columns=["Unnamed: 0"])

    with open(VOCAB_PATH) as f:
        words = json.load(f)
    words = pd.DataFrame.from_dict(words, orient='index', columns=["idx"]).reset_index()
    words.drop(columns=["idx"], inplace=True)
    words.rename(columns={"index":"label"}, inplace=True)

    IX_TO_WORD = create_ix_to_vocab_map(VOCAB_PATH)

    VOCAB_SIZE = len(words)

    en_idx = [0, 1, 2, 6, 18, 19] + list(range(37, VOCAB_SIZE))
    fr_idx = list(range(37))

    {"<pad>": 0, "<cls>": 1, "<eos>": 2, "allez": 3, "de": 4, "cinquante": 5, "-": 6, "huit": 7, "\u00e0": 8, "droite": 9, "soixante": 10, "et": 11, "onze": 12, "gauche": 13, "puis": 14, "descendez": 15, "quatre": 16, "vingt": 17, "six": 18, ",": 19, "montez": 20, "quinze": 21, "trente": 22, "un": 23, "douze": 24, "neuf": 25, "quarante": 26, "dix": 27, "deux": 28, "sept": 29, "quatorze": 30, "vingts": 31, "cinq": 32, "trois": 33, "treize": 34, "seize": 35, "cent": 36, "move": 37, "forty": 38, "eight": 39, "to": 40, "the": 41, "left": 42, "then": 43, "go": 44, "down": 45, "ten": 46, "up": 47, "eleven": 48, "seventy": 49, "right": 50, "and": 51, "twenty": 52, "thirty": 53, "seven": 54, "sixty": 55, "five": 56, "nineteen": 57, "one": 58, "twelve": 59, "fifty": 60, "nine": 61, "eighty": 62, "three": 63, "ninety": 64, "two": 65, "seventeen": 66, "sixteen": 67, "four": 68, "fourteen": 69, "eighteen": 70, "fifteen": 71, "hundred": 72, "thirteen": 73}

    # Load Model
    # Specifies the device, language, model type, and number of epochs, then loads in each checkpoint.
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    MODEL_CHECKPOINTS = ["../model/saved_models/{}_{}/model_epoch_{}.pt".format(
        LANG, model_type, i) for i in range(1, N)]
    MODELS = [torch.load(checkpoint, map_location=device) for checkpoint in MODEL_CHECKPOINTS]
    
    # Get pd dataframe of embedding for each word
    EMBEDS = [list(model.children())[:-1][0] for model in MODELS]
    embed = EMBEDS[-1]
    to_embed = torch.tensor(range(VOCAB_SIZE), dtype=torch.long, device=device)
    embeddings = embed(to_embed).cpu().detach().numpy()
    words = pd.concat([words, pd.DataFrame(embeddings)], axis=1)


    # SPLIT DATASET INTO ENGLISH/FRENCH
    to_pca = words[words.columns.tolist()[1:]]
    to_pca_en = to_pca.iloc[en_idx, :]
    to_pca_fr = to_pca.iloc[fr_idx, :]


    # PCA
    pcaed_en = PCA_(2, to_pca_en)
    pcaed_fr = PCA_(2, to_pca_fr)
    plot_embeds(pcaed_en, list(words.iloc[en_idx,:]['label']), title="PCA Embeddings English")
    plot_embeds(pcaed_fr, words.iloc[fr_idx,:]['label'], title="PCA Embeddings French")

    # TSNE
    tsed_en = tSNE(to_pca_en)
    tsned_fr = tSNE(to_pca_fr)
    plot_embeds(tsed_en, list(words.iloc[en_idx,:]['label']), title="tSNE Embeddings English")
    plot_embeds(tsned_fr, words.iloc[fr_idx,:]['label'], title="tSNE Embeddings French")




if __name__=="__main__":
    main()