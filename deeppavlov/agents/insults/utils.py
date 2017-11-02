"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os, sys
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import scipy.sparse as sp


def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors.

    Args:
        opt: dictionary of given parameters
        word_dict: dictionary of words

    Returns:
        embedding matrix
    """
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
    """Create and save vectorizers and feature selectors on given train data.

    Args:
        train_data: list of train text samples
        train_labels: list of train labels
        ind: index of vectorizer/selector to save file
        model_file: model filename
        ngram_range_: range of n-grams
        max_num_features: maximum number of features to select
        analyzer_type: analyzer type for TfidfVectorizer 'word' or 'char'

    Returns:
        nothing
    """
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
    """Extract special features from data corresponding to "you are" experssion.

    Args:
        data: list of text samples

    Returns:
        list of special expressions
    """
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
                               ngram_list=[1], max_num_features_list=[100],
                               analyzer_type_list=['word']):
    """Call creation and save of vectorizers and selectors including special cases.

    Args:
        train_data: list of train text samples
        train_labels:  list of train labels
        model_file: model filename
        ngram_list: list of ranges of n-grams
        max_num_features_list: list of maximum number of features to select
        analyzer_type_list: list of analyzer types for TfidfVectorizer 'word' or 'char'

    Returns:
        nothing
    """
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
    """Read vectorizers and selectors from file.

    Args:
        model_file: model filename
        num_ngrams: number of different n-grams considered

    Returns:
        list of vectorizers, list of selectors
    """
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
    """Vectorize data and select features.

    Args:
        data: list of text train samples
        vectorizers: list of vectorizers
        selectors: list of selectors

    Returns:
        features extracted from data using vectorizers and selectors lists
    """
    num_ngrams = len(vectorizers) - 1
    x = None

    for i in range(num_ngrams):
        x_i = vectorizers[i].transform(data)
        if selectors[i] is not None:
            x_i = selectors[i].transform(x_i)
        if i == 0:
            x = x_i
        else:
            x = sp.hstack([x, x_i])

    data_special = ngrams_you_are(data)
    x_i = vectorizers[-1].transform(data_special)
    if selectors[-1] is not None:
        x_i = selectors[-1].transform(x_i)
    x = sp.hstack([x, x_i])
    return x
