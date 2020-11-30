import os
import pandas as pd
from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import Lowercase, StripAccents, Replace
from tokenizers.models import WordPiece, BPE
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Code written with the help of Huggingface's tutorial:
# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#build-a-tokenizer-from-scratch

def read_cfg_train_data(filename, header=True):
    '''
    Given training data for decoder (e.g. en.csv or fr.csv), outputs
    the list of samples to tokenize as a list (try generator too)
    '''
    return list(pd.read_csv(filename)['string'])


###### Functions for training tokenizern ######

def train_tokenizer(corpora, model='wp', vocab_size=50,
                    save=True, filename='wp_model.json', overwrite=False):
    '''
    Given a set of training corpora, trains a tokenizer to tokenize
    See wordpiece method: https://huggingface.co/transformers/tokenizer_summary.html

    @:param corpora: a list of files (or lists) containing the corpus to train on
            model: either 'wp' or 'bpe' meaning word piece or byte-pair encoding
            vocab size (int): the desired vocabulary size
            save (bool): whether or not to save the tokenizer model
            filename (str): filename of the saved model
            overwrite (bool): if True, will overwrite if finds a file with the same name

    @:returns: trained tokenizer model
    '''
    str_to_tokenizer = {'wp': (WordPiece, WordPieceTrainer),
                        'bpe': (BPE, BpeTrainer)} # map model input st to model obj
    # init tokenizer

    model_ = str_to_tokenizer[model][0]
    # tokenizer = Tokenizer(model_({"[UNK]":0}))
    tokenizer = Tokenizer(model_())
    # choose word piece or bpe as trainer
    trainer = str_to_tokenizer[model][1](vocab_size=vocab_size,
                                         special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])

    # make lowercase, pre-tokenize, then tokenize according to model arg
    tokenizer.normalizer = normalizers.Sequence([Replace(str(i), '') for i in range(10)] # workout for regex not working to replace digits
                                                + [Replace(',"', '"'), # replace commas before and after message
                                                Replace('",', '"'), Lowercase()])

    # note - this normalizer still leaves a weird comma at the end. Needs to be removed
    tokenizer.pre_tokenizer = Whitespace() # removes white spaces
    tokenizer.train(trainer, corpora) #, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]) # TODO - can corpora come as lists?

    # init UNK tokens by saving, then loading (this is required in this edition for some reason)

    # tokenizer.save(os.path.join('models', filename + '_no_unk'))
    # tokenizer.model = str_to_tokenizer[model][0].from_file(os.path.join('models', filename + '_no_unk'), unk_token="[UNK]")

    # save model
    if save:
        # raise warning if model already there
        if os.path.isdir(os.path.join('models', filename)) and not overwrite:
            raise FileExistsError('You are attempting to overwrite an existing file. Please use a different \
                                  file name to save this model or set overwrite=True to overwrite')
            return
        else:
            tokenizer.save(os.path.join('../models', filename))

    return tokenizer


def load_tokenizer_from_file(filename, model_type='wp'):
    '''
    Load previously-trained and saved tokenizer
    @:param filename : the file where the tokenizer is saved
    @:returns trained tokenizer model
    '''
    return Tokenizer.from_file(filename) # doesn't take unk_token parameter

    # TODO - consider investigating this further
    # if model_type == 'wp':
    #     return WordPiece.from_file(filename, unk_token="[UNK]")
    # elif model_type == 'bpe':
    #     return BPE.from_file(filename, unk_token="[UNK]")
    # else:
    #     raise ValueError(f'No model of type {model_type}. Use "wp" or "bpe"')
    # TODO - if getting weird unknown token errors later, maybe add this arg unk_token="[UNK]") to .fromfile()


def add_terminals_to_token(token):
    '''
    Given a token, replaces terminating quotes " " with <cls> or <eos>
    '''
    token = ['<cls>'] + token + ['<eos>']
    return token


def encode_batch(tokenizer_, batch):
    '''
    Tokenizes a set of input training data
    @:param
            batch: list of text sequences to encode
            tokenizer_ : a trained tokenizer_model
    @:returns:
        the encoded version of a batch of text sequences

    '''
    return tokenizer_.encode_batch(batch)


def decode_batch(encoded_batch, tokenizer_):
    '''
    :param encoded_batch: list of ids
    :param tokenizer_: tokenizer model
    :return:
        the decoded version of a batch of ids
    '''
    return tokenizer_.decode_batch(encoded_batch)


if __name__ == "__main__":
    print('running tokenizer')
    tokenizer = train_tokenizer(['../../generate-data/data/train/en.csv',
                    '../../generate-data/data/train/fr.csv'],
                    model='wp',
                    vocab_size=120,
                    filename='wp_model.json') # c


    # # load save tokenizer
    tokenizer = load_tokenizer_from_file(filename='../models/wp_model.json')

    # testing here :
    batch = ['descendez', 'allez', 'gauche', 'la droite', 'cinquante', 'fifty',
             'quatre-vingts dix-sept', 'fifteen',
             'Allez de quatre à gauche, et puis montez de cinquante-quatre',
             'allez de soixante à droite, and then montez vingt et un',
             'I really wonder how well the tokenizer will work']

    print(f'tokenizer: {dir(tokenizer)}')
    encoded_batch = encode_batch(tokenizer, batch)
    for seq, enc in zip(batch, encoded_batch):
        print(f'Seq: {seq} \nEnc: {enc.tokens}\n')
