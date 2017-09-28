import collections
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment
import operator
import copy
#  evalution metrics
"""
Mostly borrowed from https://github.com/clarkkev/deep-coref/blob/master/evaluation.py
"""


def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v

class RetrievalEvaluator(object):
  def __init__(self):
    self._num_correct = 0
    self._num_gold = 0
    self._num_predicted = 0

  def update(self, gold_set, predicted_set):
    self._num_correct += len(gold_set & predicted_set)
    self._num_gold += len(gold_set)
    self._num_predicted += len(predicted_set)

  def recall(self):
    return maybe_divide(self._num_correct, self._num_gold)

  def precision(self):
    return maybe_divide(self._num_correct, self._num_predicted)

  def metrics(self):
    recall = self.recall()
    precision = self.precision()
    f1 = maybe_divide(2 * recall * precision, precision + recall)
    return recall, precision, f1


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem



def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def flatten(l):
  return [item for sublist in l for item in sublist]

def load_char_dict(char_vocab_path):
  vocab = [u"<unk>"]
  with open(char_vocab_path) as f:
    vocab.extend(c.strip() for c in f.readlines())
  char_dict = collections.defaultdict(int)
  char_dict.update({c: i for i, c in enumerate(vocab)})
  return char_dict

def load_embedding_dict(embedding_path, embedding_size, embedding_format):
  print("Loading word embeddings from {}...".format(embedding_path))
  default_embedding = np.zeros(embedding_size)
  embedding_dict = collections.defaultdict(lambda: default_embedding)
  skip_first = embedding_format == "vec"
  with open(embedding_path) as f:
    for i, line in enumerate(f.readlines()):
      if skip_first and i == 0:
        continue
      splits = line.split()
      assert len(splits) == embedding_size + 1
      word = splits[0]
      embedding = np.array([float(s) for s in splits[1:]])
      embedding_dict[word] = embedding
  print("Done loading word embeddings.")
  return embedding_dict

def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def projection(inputs, output_size, initializer=None):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

#  Networks
def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 2:
    current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
  else:
    current_inputs = inputs

  for i in range(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
    current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size])
  outputs = tf.matmul(current_inputs, output_weights) + output_bias

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
  elif len(inputs.get_shape()) > 3:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
  return outputs

def cnn(inputs, filter_sizes, num_filters):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}".format(i)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, dropout):
    self._num_units = num_units
    self._dropout = dropout
    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
    initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  def preprocess_input(self, inputs):
    return projection(inputs, 3 * self.output_size)

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      c, h = state
      h *= self._dropout_mask
      projected_h = projection(h, 3 * self.output_size, initializer=self._initializer)
      concat = inputs + projected_h
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer

#

class DocumentState(object):
    def __init__(self):
        self.doc_key = None
        self.text = []
        self.text_speakers = []
        self.speakers = []
        self.sentences = []
        self.clusters = collections.defaultdict(list)
        self.stacks = collections.defaultdict(list)

    def assert_empty(self):
        assert self.doc_key is None
        assert len(self.text) == 0
        assert len(self.text_speakers) == 0
        assert len(self.sentences) == 0
        assert len(self.speakers) == 0
        assert len(self.clusters) == 0
        assert len(self.stacks) == 0

    def assert_finalizable(self):
        assert self.doc_key is not None
        assert len(self.text) == 0
        assert len(self.text_speakers) == 0
        assert len(self.sentences) > 0
        assert len(self.speakers) > 0
        assert all(len(s) == 0 for s in self.stacks.values())

    def finalize(self):
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = flatten(merged_clusters)
        # print len(all_mentions), len(set(all_mentions))

        if len(all_mentions) != len(set(all_mentions)):
            c = Counter(all_mentions)
            for x in c:
                if c[x] > 1:
                    z = x
                    break
            for i in range(len(all_mentions)):
                if all_mentions[i] == z:
                    all_mentions.remove(all_mentions[i])
                    break
        assert len(all_mentions) == len(set(all_mentions))

        return {
            "doc_key": self.doc_key,
            "sentences": self.sentences,
            "speakers": self.speakers,
            "clusters": merged_clusters
        }

def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def conll2modeldata(data):
    document_state = DocumentState()
    document_state.assert_empty()
    document_state.doc_key = "{}_{}".format(data['doc_id'][0], int(data['part_id'][0]))
    for i in range(len(data['doc_id'])):
        word = normalize_word(data['word'][i])
        coref = data['coreference'][i]
        speaker = data['speaker'][i]
        word_index = i + 1
        document_state.text.append(word)
        document_state.text_speakers.append(speaker)

        if (data['part_of_speech'][i] == 'SEN'):
            document_state.sentences.append(tuple(document_state.text))
            del document_state.text[:]
            document_state.speakers.append(tuple(document_state.text_speakers))
            del document_state.text_speakers[:]
            pass
        else:
            if coref == "-":
                pass
            else:
                for segment in coref.split("|"):
                    if segment[0] == "(":
                        if segment[-1] == ")":
                            cluster_id = int(segment[1:-1])
                            document_state.clusters[cluster_id].append((word_index, word_index))
                        else:
                            cluster_id = int(segment[1:])
                            document_state.stacks[cluster_id].append(word_index)
                    else:
                        cluster_id = int(segment[:-1])
                        start = document_state.stacks[cluster_id].pop()
                        document_state.clusters[cluster_id].append((start, word_index))

    document_state.assert_finalizable()
    return document_state.finalize()

def output_conll(input_file, predictions):
    prediction_map = {}
    output_file = copy.deepcopy(input_file)

    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k, v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k, v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    for i in range(len(input_file['doc_id'])):
        doc_key = '{}_{}'.format(input_file['doc_id'], input_file['part_id'])
        start_map, end_map, word_map = prediction_map[doc_key]
        coref_list = []
        if word_index in end_map:
            for cluster_id in end_map[word_index]:
                coref_list.append("{})".format(cluster_id))
        if word_index in word_map:
            for cluster_id in word_map[word_index]:
                coref_list.append("({})".format(cluster_id))
        if word_index in start_map:
            for cluster_id in start_map[word_index]:
                coref_list.append("({}".format(cluster_id))

        if len(coref_list) == 0:
            output_file['coreference'][i] = "-"
        else:
            output_file['coreference'][i] = "|".join(coref_list)

        word_index += 1
    return output_file