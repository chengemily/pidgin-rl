import os
import json

from preprocess import tokenize_
from preprocess import word_embed


def main():
    # TODO - make these arg parse arguments?
    en_corpus = '../generate-data/data_final/train/en.csv'
    fr_corpus = '../generate-data/data_final/train/fr.csv' 
    save_dir = '../decoder/data_final'
    embed_dir = '../decoder/models'
    vocab_size = 300
    vector_size = 20

    # Load tokenizer or train if not present
    print('Training tokenizer')
    #if os.path.isfile('models/wp_model_words.json'):
     #   tokenizer = tokenize_.load_tokenizer_from_file(filename='models/wp_model_words.json')
    #else:
    tokenizer = tokenize_.train_tokenizer([en_corpus, fr_corpus],
                    vocab_size=vocab_size) # c

    print(tokenizer)
    # read in data and encode into tokens
    print('Encoding Tokens')
    fr_data = tokenize_.read_cfg_train_data(fr_corpus)
    en_data = tokenize_.read_cfg_train_data(en_corpus)

    enc_fr = tokenize_.encode_batch(tokenizer, fr_data)
    enc_en = tokenize_.encode_batch(tokenizer, en_data)
    encodings = enc_fr + enc_en
    tokens = [tokenize_.add_terminals_to_token(enc.tokens) for enc in encodings]
    word_to_ix, ix_tokens = word_embed.transform_tokens_to_ix(tokens)

    ix_tokens_fr = ix_tokens[:len(enc_fr)]
    ix_tokens_en = ix_tokens[len(enc_fr):]

    print(f'french tokens : {ix_tokens_fr}')
    print(f'english tokens : {ix_tokens_en}')

    # train word2vec model on concat of french and english corpora
    print('Training word2vec model')
    # if os.path.isfile('models/embed_model.json'):
    #     embedder = word_embed.load_word2vec_model('../models/embed_model.json')
    # else:
    embedder = word_embed.train_word2vec_model(ix_tokens,
                                               filename=os.path.join(embed_dir, 'embed_model_words.json'),
                                               window=3,
                                               size=vector_size)

    print('Embedding all words')
    # embed all words, save as dict
    word_embeddings = {}
    # create lookup for word to vector
    for sentence in ix_tokens:
        for word in sentence:
            word_embeddings[word] = embedder.wv[word].tolist() # saving as list so json doesn't get angry

    # Sentence indices
    sentence_i_fr = [[word_to_ix[word] for word in sentence] for sentence in ix_tokens_fr]
    sentence_i_en = [[word_to_ix[word] for word in sentence] for sentence in ix_tokens_en]

    # Embed each of the token sentences
    sentence_embeddings_fr = []
    for sentence in ix_tokens_fr:
        sentence_embeddings_fr.append([word_embeddings[word] for word in sentence])

    sentence_embeddings_en = []
    for sentence in ix_tokens_en:
        sentence_embeddings_en.append([word_embeddings[word] for word in sentence])

    # save everything as json
    print('Saving to json')


    # dump vocabulary
    with open(os.path.join(save_dir, 'vocab_words.json'), 'w') as f:
        json.dump(word_to_ix, f)

    # dump indexed data
    with open(os.path.join(save_dir, 'tokens_words.json'), 'w') as f:
        json.dump({'fr': ix_tokens_fr, 'en': ix_tokens_en}, f)

    # dumpindexed data
    with open(os.path.join(save_dir, 'indexed_data_words.json'), 'w') as f:
        json.dump({'fr': sentence_i_fr, 'en': sentence_i_en}, f)

    # dump embedded data
    with open(os.path.join(save_dir, 'embedded_data_words.json'), 'w') as f:
        json.dump({'fr': sentence_embeddings_fr, 'en': sentence_embeddings_en}, f)

    # dump word embeddings
    with open(os.path.join(save_dir, 'embeddings_words.json'), 'w') as outfile:
        json.dump({'word_to_vec': word_embeddings}, outfile)



if __name__ == "__main__":
    main()
