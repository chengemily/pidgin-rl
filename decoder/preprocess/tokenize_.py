import os

from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import Lowercase, StripAccents, Replace
from tokenizers.models import WordPiece, BPE
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace





def read_cfg_train_data(filename):
    '''
    Given training data for decoder (e.g. en.csv or fr.csv), outputs
    the list of samples to tokenize as a list (try generator too)
    '''
    with open(filename, 'r') as file:
        data = file.read()
    return data


###### Functions for training tokenizern ######

def train_tokenizer(corpora, model='wp', vocab_size=30000,
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
    tokenizer = Tokenizer(str_to_tokenizer[model][0]())
    # choose word piece or bpe as trainer
    trainer = str_to_tokenizer[model][1](vocab_size=vocab_size,
                                         special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # make lowercase, pre-tokenize, then tokenize according to model arg
    tokenizer.normalizer = normalizers.Sequence([Replace(str(i), '') for i in range(10)] # workout for regex not working to replace digits
                                                 + [Replace('-', ''), Lowercase()])
                                                 #Replace('(-)*[0-9]+', '')]) # TODO - double check if we want to replace digits (+/-)
    tokenizer.pre_tokenizer = Whitespace() # removes white spaces
    tokenizer.train(trainer, corpora) # TODO - can corpora come as lists?

    # save model
    if save:
        # raise warning if model already there
        if os.path.isdir(os.path.join('models', filename)) and not overwrite:
            raise FileExistsError('You are attempting to overwrite an existing file. Please use a different \
                                  file name to save this model or set overwrite=True to overwrite')
            return
        else:
            tokenizer.save(os.path.join('models', filename))

    return tokenizer


def load_tokenizer_from_file(filename):
    '''
    Load previously-trained and saved tokenizer
    @:param filename : the file where the tokenizer is saved
    @:returns trained tokenizer model
    '''
    return Tokenizer.from_file(filename)
    # TODO - if getting weird unknown token errors later, maybe add this arg unk_token="[UNK]") to .fromfile()



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
    # tokenizer = train_tokenizer(['../generate-data/data/train/en.csv',
    #                 '../generate-data/data/train/fr.csv'],
    #                 vocab_size=1000) # c
    #
    #
    # # # load save tokenizer
    # tokenizer = load_tokenizer_from_file(filename='models/wp_model.json')
    #
    # # testing here :
    # batch = ['Move Three to the right and then four up, -5, 10',
    #          'Allez de quatre à gauche, et puis montez de cinquante',
    #          'Montez de ten to the left, et puis deux to the right',
    #          'what about this random sentence',
    #          'allez down soisante, and then montez vingte et un',
    #          'I really wonder how well the tokenizer will work']
    #
    # print(f'tokenizer: {dir(tokenizer)}')
    # encoded_batch = encode_batch(tokenizer, batch)
    # for seq, enc in zip(batch, encoded_batch):
    #     print(f'Seq: {seq} \nEnc: {enc.tokens}\n')