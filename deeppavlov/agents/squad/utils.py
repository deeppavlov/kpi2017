# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import time
import unicodedata
import numpy as np
from collections import Counter


# ------------------------------------------------------------------------------
# Data/model utilities.
# ------------------------------------------------------------------------------

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors."""
    embeddings = np.random.normal(0.0, 1.0, (len(word_dict), opt['word_embedding_dim']))

    # Fill in embeddings
    if not opt.get('embedding_file'):
        raise RuntimeError('Tried to load embeddings with no embedding file.')
    with open(opt['embedding_file']) as f:
        for line in f:
            parsed = line.rstrip().split(' ')
            if len(parsed)==2:
                continue
            assert(len(parsed) == opt['word_embedding_dim'] + 1)
            w = normalize_text(parsed[0])
            if w in word_dict:
                vec = np.array([float(i) for i in parsed[1:]])
                embeddings[word_dict[w]] = vec

    # Zero NULL token
    embeddings[word_dict['__NULL__']] = np.zeros(opt['word_embedding_dim'])

    return embeddings

def build_feature_dict(opt):
    """Make mapping of feature option to feature index."""
    feature_dict = {}
    if opt['use_in_question']:
        feature_dict['in_question'] = len(feature_dict)
        feature_dict['in_question_uncased'] = len(feature_dict)
    if opt['use_tf']:
        feature_dict['tf'] = len(feature_dict)
    if opt['use_time'] > 0:
        for i in range(opt['use_time'] - 1):
            feature_dict['time=T%d' % (i + 1)] = len(feature_dict)
        feature_dict['time>=T%d' % opt['use_time']] = len(feature_dict)
    return feature_dict


# ------------------------------------------------------------------------------
# Torchified input utilities.
# ------------------------------------------------------------------------------

def embed_word(word, word_dict, embeddings):
    try:
        return embeddings[word_dict[word]]
    except:
        print('Unrecognized word!')
        return np.random.normal(0, 1, size=(embeddings[0].shape[0]))


def vectorize(opt, ex, word_dict, feature_dict, embeddings):
    """Turn tokenized text inputs into feature vectors."""
    # Index words
    document = np.array([embed_word(w, word_dict, embeddings) for w in ex['document']])
    question = np.array([embed_word(w, word_dict, embeddings) for w in ex['question']])

    # Create extra features vector
    features = np.zeros((len(ex['document']), len(feature_dict)))

    # f_{exact_match}
    if opt['use_in_question']:
        q_words_cased = set([w for w in ex['question']])
        q_words_uncased = set([w.lower() for w in ex['question']])
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0

    # f_{tf}
    if opt['use_tf']:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    if opt['use_time'] > 0:
        # Counting from the end, each (full-stop terminated) sentence gets
        # its own time identitfier.
        sent_idx = 0
        def _full_stop(w):
            return w in {'.', '?', '!'}
        for i, w in reversed(list(enumerate(ex['document']))):
            sent_idx = sent_idx + 1 if _full_stop(w) else max(sent_idx, 1)
            if sent_idx < opt['use_time']:
                features[i][feature_dict['time=T%d' % sent_idx]] = 1.0
            else:
                features[i][feature_dict['time>=T%d' % opt['use_time']]] = 1.0

    # Maybe return without target
    if ex['target'] is None:
        return document, features, question

    # ...or with target
    start = ex['target'][0]
    end = ex['target'][1]

    return document, features, question, start, end


def batchify(batch, null=0):
    """Collate inputs into batches."""
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 2

    # Get elements
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    text = [ex[-2] for ex in batch]
    spans = [ex[-1] for ex in batch]

    # Batch documents and features
    #print(docs[0].shape)
    max_length = max([d.shape[0] for d in docs])
    emb_dim = docs[0].shape[1]
    x1 = np.zeros((len(docs), max_length, emb_dim))
    x1_mask = np.ones((len(docs), max_length, emb_dim))
    x1_f = np.zeros((len(docs), max_length, features[0].shape[1]))
    for i, d in enumerate(docs):
        x1[i, :d.shape[0], :] = d
        x1_mask[i, :d.shape[0], :] = 0
        x1_f[i, :d.shape[0]] = features[i]

    # Batch questions
    max_length = max([q.shape[0] for q in questions])
    x2 = np.zeros((len(questions), max_length, emb_dim))
    x2_mask = np.ones((len(questions), max_length, emb_dim))
    for i, q in enumerate(questions):
        x2[i, :q.shape[0], :] = q
        x2_mask[i, :q.shape[0]] = 0

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x2, x2_mask, text, spans

    # ...Otherwise add targets
    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        y_s = np.hstack([ex[3] for ex in batch])
        y_e = np.hstack([ex[4] for ex in batch])
        return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, text, spans

    # ...Otherwise wrong number of inputs
    raise RuntimeError('Wrong number of inputs per batch')


# ------------------------------------------------------------------------------
# General logging utilities.
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
