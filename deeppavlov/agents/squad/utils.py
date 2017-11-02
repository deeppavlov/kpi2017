# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import time
import unicodedata
import numpy as np
from numpy.random import seed
import re
import string
from collections import Counter
from keras.optimizers import Adam, Adamax, Adadelta

# ------------------------------------------------------------------------------
# Optimizer presets.
# ------------------------------------------------------------------------------
def getOptimizer(optim, exp_decay, grad_norm_clip, lr = 0.001):
    '''
    Function for setting up optimizer, combines several presets from
    published well performing models on SQuAD.
    '''
    optimizers = {
        'Adam': Adam(lr=lr, decay=exp_decay, clipnorm=grad_norm_clip),
        'Adamax': Adamax(lr=lr, decay=exp_decay, clipnorm=grad_norm_clip),
        'Adadelta': Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, decay=exp_decay, clipnorm=grad_norm_clip)
    }

    try:
        optimizer = optimizers[optim]
    except KeyError as e:
        raise ValueError('problems with defining optimizer: {}'.format(e.args[0]))

    del (optimizers)
    return optimizer

# ------------------------------------------------------------------------------
# Data/model utilities.
# ------------------------------------------------------------------------------

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors."""
    seed(1)
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
    seed(1)
    try:
        return embeddings[word_dict[word]]
    except:
        print('Unrecognized word!')
        return np.random.normal(0, 1, size=(embeddings[0].shape[0]))

def embed_index(word, word_dict):
    try:
        return word_dict[word]
    except:
        return len(word_dict)


def vectorize(opt, ex, word_dict, feature_dict, embeddings):
    """Turn tokenized text inputs into feature vectors."""

    # Old way is fo
    if not opt['inner_embeddings']:
        # Index words
        document = np.array([embed_word(w, word_dict, embeddings) for w in ex['document']])
        question = np.array([embed_word(w, word_dict, embeddings) for w in ex['question']])
    elif opt['inner_embeddings']:
        document = np.array([embed_index(w, word_dict) for w in ex['document']])
        question = np.array([embed_index(w, word_dict) for w in ex['question']])

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

    # f_{time}
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
    x1_mask = np.zeros((len(docs), max_length))
    x1_f = np.zeros((len(docs), max_length, features[0].shape[1]))
    for i, d in enumerate(docs):
        x1[i, :d.shape[0], :] = d
        x1_mask[i, :d.shape[0]] = 1.0
        x1_f[i, :d.shape[0]] = features[i]

    # Batch questions
    max_length = max([q.shape[0] for q in questions])
    x2 = np.zeros((len(questions), max_length, emb_dim))
    x2_mask = np.zeros((len(questions), max_length))
    for i, q in enumerate(questions):
        x2[i, :q.shape[0], :] = q
        x2_mask[i, :q.shape[0]] = 1.0

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


# ------------------------------------------------------------------------------
# Scoring utilities.
# ------------------------------------------------------------------------------

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100.0 * em / total
    f1 = 100.0 * f1 / total
    return em, f1
