import os
import json

from preprocess import tokenize_
from preprocess import word_embed



def main():
    # TODO - make these arg parse arguments?
    en_corpus = '../generate-data/data/train/en.csv'
    fr_corpus = '../generate-data/data/train/fr.csv'
    vocab_size = 1000

    # Load tokenizer or train if not present
    print('Training tokenizer')
    if os.path.isfile('models/wp_model.json'):
        tokenizer = tokenize_.load_tokenizer_from_file(filename='models/wp_model.json')
    else:
        tokenizer = train_tokenizer([en_corpus, fr_corpus],
                    vocab_size=vocab_size) # c

    # read in data and encode into tokens
    print('Encoding Tokens')
    fr_data = tokenize_.read_cfg_train_data(fr_corpus)
    en_data = tokenize_.read_cfg_train_data(en_corpus)
    encodings = tokenize_.encode_batch(tokenizer, fr_data) + tokenize_.encode_batch(tokenizer, en_data)
    tokens = [enc.tokens for enc in encodings]
    # print(f'tokens: {token.}')

    # train word2vec model on concat of french and english corpora
    print('Training word2vec model')
    if os.path.isfile('models/embed_model.json'):
        embedder = word_embed.load_word2vec_model('models/embed_model.json')
    else:
        embedder = word_embed.train_word2vec_model(tokens,
                                                   filename='models/embed_model.json',
                                                   window=3)

    print('Embedding all words')
    # embed all words, save as dict
    word_embeddings = {}
    for sentence in tokens:
        for word in sentence:
            word_embeddings[word] = embedder[word].tolist() # saving as list so json doesn't get angry

    print(word_embeddings)

    # save everything as json
    print('Saving to json')
    data = {}
    data['embeddings'] = word_embeddings
    with open('data/embeddings.json', 'w') as outfile:
        json.dump(data, outfile)



if __name__ == "__main__":
    main()
