import nltk
import gensim
from gensim.models import Word2Vec
import torch
from torch.nn.utils.rnn import pad_sequence


def transform_tokens_to_ix(tokens):
    '''
    Given a list of tokens, constructs a map from word: ix
    :param tokens: list of tokens
    :return: vocab_map and indexed tokens
    '''
    vocab_map = {'<pad>': 0,
                 '<cls>': 1,
                 '<eos>': 2}

    ix_map = {0: '<pad>',
              1: '<cls>',
              2: '<eos>'}

    ix_tokens = []
    # TODO - make this more efficient with itertools?
    ix = 2
    for token in tokens:
        new_token = []
        for word in token:
            if word not in vocab_map:
                ix += 1
                vocab_map[word] = ix
                ix_map[ix] = word
                new_token.append(ix)
            else:
                new_token.append(vocab_map[word])
        ix_tokens.append(torch.tensor(new_token))

    # Pad sequence, map to tokens
    ix_tokens_tensor = pad_sequence(ix_tokens).T # pad
    ix_tokens_int = ix_tokens_tensor.tolist() #make to list
    ix_tokens_final = [[ix_map[i] for i in token] for token in ix_tokens_int] # back to tokens
    return vocab_map, ix_tokens_final # transpose to get samples by max_seq length



def train_word2vec_model(data,
                         save=True,
                         filename='models/embed_model.json',
                         **kwargs):
    '''
    Given a tokenized corpus data, trains a word2vec model with
    specified parameters in **kwargs
    :param  data:
            kwargs: feed into gensim.models.word2vec
    :return: trained Word2Vec model
    '''
    model = Word2Vec(data, min_count=1, **kwargs) # inits and trains word2vec model
    if save:
        model.save(filename)
    return model


def load_word2vec_model(filename):
    '''
    Loads word2vec model from filename
    :param filename:
    :return:
    '''
    return Word2Vec.load(filename)


if __name__ == "__main__":
    print('running word_embed.py')