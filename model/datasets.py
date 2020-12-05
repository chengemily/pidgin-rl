import json
import pandas as pd
import torch
import numpy as np

def load_json(path):
    """
    Load pretrained embeddings as dict
    :param vectors_path: (str)
    :return:
    """
    with open(path) as f:
        data = json.load(f)
    return data


def preprocess_data(data_path, embeds_path, lang='fr'):
    """
    Loads pre-embedded dataset and labels, in a random (but consistent) order.
    :param data_path: (str) filepath to csv
    :param embeds_path: (str) filepath to json
    :return: X (list of list of list), y (len(train) x 2) np array
    """
    X = load_json(embeds_path)[lang] # list (dataset) of list (command) of list (word embedding)
    data = pd.read_csv(data_path)
    X_str = data['string'].tolist()
    y = data[['x', 'y']].values / 100

    return X_str, X, y


def train_test_split(X, y, spl = 0.8):
    """

    :param X:
    :param y:
    :return: trainX, trainY, testX, testY
    """
    train_idx = round(spl * len(y))
    return X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]


def batch_data(n, lst, device):
    """
    Splits lst into batches of constant size and convert to tensor.
    :param n:
    :param lst:
    :return: tensor
    """
    return [torch.tensor(lst[i: min(i + n, len(lst))]).to(device) for i in range(0, len(lst), n)][:-1]


def load_data(data_path, embeds_path, lang, batch_size, device):
    """

    :param data_path:
    :param embeds_path:
    :param lang: (str) fr or en
    :param batch_size: (int)
    :return: train_batched = (trainX, trainy), test_batched = (testX, testy)
    """
    print("Loading data...")
    X_str, X, y = preprocess_data(data_path, embeds_path, lang=lang)
    Xtrain, ytrain, Xtest, ytest = train_test_split(X, y)
    Xtrain_len, Xtest_len = [sum(length > 0 for length in sentence) for sentence in Xtrain], [sum(length > 0 for length in sentence) for sentence in Xtest]
    Xtrain_batched = batch_data(batch_size, Xtrain, device)
    Xtest_batched = batch_data(batch_size, Xtest, device)
    Xtrain_len_batched = batch_data(batch_size, Xtrain_len, device)
    Xtest_len_batched = batch_data(batch_size, Xtest_len, device)
    ytrain_batched = batch_data(batch_size, ytrain, device)
    ytest_batched = batch_data(batch_size, ytest, device)

    return X_str, (Xtrain_batched, Xtrain_len_batched, ytrain_batched), (Xtest_batched, Xtest_len_batched, ytest_batched)


# Functions for translating indexes and words

def create_ix_to_vocab_map(filename='../tokenizer/data/vocab.json'):
    """
    Creates ix: word dict

    :param: filename is lcoation of vocab.json
    :returns: dictionary of ix_to_word
    """
    data = load_json(filename)
    ix_to_word = {v:k for k,v in data.items()}
    return ix_to_word

def translate_sentence_ix_to_word(vec, ix_to_word):
    '''
    :param vec: Vector of indexes (ints)
    :param ix_to_word: dictionary of ix(int): word(str)
    :return: vec of words representing a sentence
    '''
    return [ix_to_word[ix.item()] for ix in vec]

def get_ix_from_softmax(batch):
    '''
    given a batch of softmax vectors, gets the argmax index to predict each word
    :param batch: tensor of dims (batch x vocab_size x target_length)
    :return: tensor (batch x sentence length)
    '''
    topv, topi = batch.topk(1, dim=1)  # taking argmax
    return topi.squeeze()


def translate_batch(batch, ix_to_word):
    '''
    :param batch: tensor(batch x vocab_length, sentence length) given a batch of indexes, translates to words
    :param ix_to_word:
    :return: list of lists, each internal list representing a sentence
    '''
    batch_ix = get_ix_from_softmax(batch)
    translated_batch = []
    for sent in batch_ix:
        translated_batch.append(translate_sentence_ix_to_word(sent, ix_to_word))
    return translated_batch






if __name__ == "__main__":
    datas = '../generate-data/data/train/fr.csv'
    embeds_path = 'embeddings.json'
    prep_data(datas, embeds_path)



