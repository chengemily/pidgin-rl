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

# TODO: change architecture to weight matrix, save tokenizer and load
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
    y = data[['x', 'y']].values

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
    Xtrain_batched = batch_data(batch_size, Xtrain, device)
    Xtest_batched = batch_data(batch_size, Xtest, device)
    ytrain_batched = batch_data(batch_size, ytrain, device)
    ytest_batched = batch_data(batch_size, ytest, device)

    return X_str, (Xtrain_batched, ytrain_batched), (Xtest_batched, ytest_batched)


if __name__ == "__main__":
    datas = '../generate-data/data/train/fr.csv'
    embeds_path = 'embeddings.json'
    prep_data(datas, embeds_path)



