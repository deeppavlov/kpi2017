import os, sys
import numpy as np
import _pickle as cPickle
import scipy.sparse as sp
sys.path.append(os.path.abspath(os.path.join("..", "..", "parlai_tasks", "insults")))
from build import ngrams_you_are

def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors."""
    embeddings_index = {}

    if not opt.get('embedding_file'):
        print('WARNING: No embeddings file set, turning trainable embeddings on')
        return None
    # Fill in embeddings
    embedding_file = os.path.join(opt['datapath'], 'paraphrases', opt.get('embedding_file'))
    if not os.path.isfile(embedding_file):
        print('WARNING: Tried to load embeddings with no embedding file, turning trainable embeddings on')
        return None
    with open(embedding_file) as f:
        for line in f:
            values = line.rsplit(sep=' ', maxsplit=opt['embedding_dim'])
            assert(len(values) == opt['embedding_dim'] + 1)
            word = values[0]
            coefs = np.asarray(values[1:-1], dtype='float32')
            embeddings_index[word] = coefs

    # prepare embedding matrix
    embedding_matrix = np.zeros((len(word_dict) + 1, opt['embedding_dim']))
    for word, i in word_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def vectorize_select(data, dpath, num_ngrams):
    for i in range(num_ngrams):
        with open(os.path.join(dpath, 'ngrams_vect_general_' + str(i) + '.bin'), 'rb') as f:
            data_struct = cPickle.load(f)
            try:
                vectorizer, selector = data_struct['vectorizer'], data_struct['selector']
                X_i = vectorizer.transform(data)
                X_i = selector.transform(X_i)
            except KeyError:
                vectorizer = data_struct['vectorizer']
                X_i = vectorizer.transform(data)
        if i == 0:
            X = X_i
        else:
            X = sp.hstack([X, X_i])

    with open(os.path.join(dpath, 'ngrams_vect_special.bin'), 'rb') as f:
        data_struct = cPickle.load(f)
        you_are_data = ngrams_you_are(data)
        try:
            vectorizer, selector = data_struct['vectorizer'], data_struct['selector']
            X_i = vectorizer.transform(you_are_data)
            X_i = selector.transform(X_i)
        except KeyError:
            vectorizer = data_struct['vectorizer']
            X_i = vectorizer.transform(you_are_data)
    X = sp.hstack([X, X_i])

    return X