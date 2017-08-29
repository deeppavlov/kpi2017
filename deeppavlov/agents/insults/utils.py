import os, sys
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import scipy.sparse as sp

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


def ngrams_selection(train_data, train_labels, ind, model_file,
                     ngram_range_=(1, 1), max_num_features=100,
                     analyzer_type='word'):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range_, sublinear_tf=True, analyzer=analyzer_type)

    X_train = vectorizer.fit_transform(train_data)

    if max_num_features < X_train.shape[1]:
        ch2 = SelectKBest(chi2, k=max_num_features)
        ch2.fit(X_train, train_labels)
        data_struct = {'vectorizer': vectorizer, 'selector': ch2}
        print ('creating ', model_file + '_ngrams_vect_' + ind + '.bin')
        with open(model_file + '_ngrams_vect_' + ind + '.bin', 'wb') as f:
            pickle.dump(data_struct, f)
    else:
        data_struct = {'vectorizer': vectorizer}
        print ('creating', model_file + '_ngrams_vect_' + ind + '.bin')
        with open(model_file + '_ngrams_vect_' + ind + '.bin', 'wb') as f:
            pickle.dump(data_struct, f)
    return

def ngrams_you_are(data):
    g = [x.lower()
         .replace("you are", " SSS ")
         .replace("you're", " SSS ")
         .replace(" ur ", " SSS ")
         .replace(" u ", " SSS ")
         .replace(" you ", " SSS ")
         .replace(" yours ", " SSS ")
         .replace(" u r ", " SSS ")
         .replace(" are you ", " SSS ")
         .replace(" urs ", " SSS ")
         .replace(" r u ", " SSS ").split("SSS")[1:]
         for x in data]
    f = []
    for x in g:
        fts = " "
        for y in x:
            w = y.strip().replace("?",".").split(".")
            fts = fts + " " + w[0]
        f.append(fts)
    return f

def create_vectorizer_selector(train_data, train_labels, model_file,
                               ngram_list=[1], max_num_features_list=[100], analyzer_type_list=['word']):

    for i in range(len(ngram_list)):
        ngrams_selection(train_data, train_labels, 'general_' + str(i), model_file,
                         ngram_range_=(ngram_list[i], ngram_list[i]),
                         max_num_features=max_num_features_list[i],
                         analyzer_type=analyzer_type_list[i])
    you_are_data = ngrams_you_are(train_data)
    ngrams_selection(you_are_data, train_labels, 'special', model_file,
                     ngram_range_=(1,1), max_num_features=100)
    return

def get_vectorizer_selector(model_file, num_ngrams):
    vectorizers = []
    selectors = []
    for i in range(num_ngrams):
        with open(model_file + '_ngrams_vect_general_' + str(i) + '.bin', 'rb') as f:
            data_struct = pickle.load(f)
            try:
                vectorizers.append(data_struct['vectorizer'])
                selectors.append(data_struct['selector'])
            except KeyError:
                vectorizers.append(data_struct['vectorizer'])
                selectors.append(None)

    with open(model_file + '_ngrams_vect_special.bin', 'rb') as f:
        data_struct = pickle.load(f)
        try:
            vectorizers.append(data_struct['vectorizer'])
            selectors.append(data_struct['selector'])
        except KeyError:
            vectorizers.append(data_struct['vectorizer'])
            selectors.append(None)
    return vectorizers, selectors

def vectorize_select_from_data(data, vectorizers, selectors):
    num_ngrams = len(vectorizers) - 1
    X = None

    for i in range(num_ngrams):
        X_i = vectorizers[i].transform(data)
        if selectors[i] is not None:
            X_i = selectors[i].transform(X_i)
        if i == 0:
            X = X_i
        else:
            X = sp.hstack([X, X_i])

    data_special = ngrams_you_are(data)
    X_i = vectorizers[-1].transform(data_special)
    if selectors[-1] is not None:
        X_i = selectors[-1].transform(X_i)
    X = sp.hstack([X, X_i])

    return X
