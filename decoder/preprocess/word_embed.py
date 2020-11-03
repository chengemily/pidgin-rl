import nltk
import gensim
from gensim.models.word2vec import Word2Vec



def train_word2vec_model(data,
                         save=True,
                         filename='models/embed_model.json' ,
                         **kwargs):
    '''
    Given a tokenized corpus data, trains a word2vec model with
    specified parameters in **kwargs
    :param  data:
            kwargs: feed into gensim.models.word2vec
    :return: trained Word2Vec model
    '''
    model = Word2Vec(data, **kwargs)
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