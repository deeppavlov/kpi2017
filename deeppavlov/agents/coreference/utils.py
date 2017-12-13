# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import collections
import tensorflow as tf
import operator
import fasttext

seed = 5
np.random.seed(seed)


def dict2conll(data, predict):
    """
    Creates conll document, and write there predicted conll string.
    Args:
        data: dict from agent with conll string
        predict: string with address and name of file

    Returns: Nothing

    """
    with open(predict, 'w') as CoNLL:
        CoNLL.write(data['conll_str'])
    return None


def normalize(v):
    """Normalize input tensor"""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v


def flatten(l):
    """Expands list"""
    return [item for sublist in l for item in sublist]


def load_char_dict(char_vocab_path):
    """Load char dict from file"""
    vocab = [u"<unk>"]
    with open(char_vocab_path) as f:
        vocab.extend(c.strip() for c in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(sorted(set(vocab)))})
    return char_dict


def load_embedding_dict(embedding_path, embedding_size, embedding_format):
    """
    Load emb dict from file, or load pre trained binary fasttext model.
    Args:
        embedding_path: path to the vec file, or binary model
        embedding_size: int, embedding_size
        embedding_format: 'bin' or 'vec'

    Returns: Embeddings dict, or fasttext pre trained model

    """
    print("Loading word embeddings from {}...".format(embedding_path))

    if embedding_format == 'vec':
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
    elif embedding_format == 'bin':
        embedding_dict = fasttext.load_model(embedding_path)
    else:
        raise ValueError('Not supported embeddings format {}'.format(embedding_format))
    print("Done loading word embeddings.")
    return embedding_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=None):
    """ Returns outputs of fully-connected network """
    return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def shape(x, dim):
    """ Returns shape of tensor """
    return x.get_shape()[dim].value or tf.shape(x)[dim]


#  Networks
def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout,
         output_weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)):
    """
    Creates fully_connected network graph with needed parameters.
    Args:
        inputs: shape of input tensor, rank of input >3 not supported
        num_hidden_layers: int32
        hidden_size: int32
        output_size: int32
        dropout: int32, dropout value
        output_weights_initializer: name of initializers or None

    Returns: network output, [output_size]

    """

    # inputs = tf.cast(inputs, tf.float32)
    if len(inputs.get_shape()) > 2:
        current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size], dtype=tf.float32)
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], dtype=tf.float32)

        current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     dtype=tf.float32)
    output_bias = tf.get_variable("output_bias", [output_size], dtype=tf.float32)
    outputs = tf.matmul(current_inputs, output_weights) + output_bias
    # outputs = tf.cast(outputs, tf.float32)
    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
    elif len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
    return outputs


def cnn(inputs, filter_sizes, num_filters):
    """
    Creates convolutional network graph with needed parameters.
    Args:
        inputs: shape of input tensor
        filter_sizes: list of shapes of filters
        num_filters: amount of filters

    Returns: network output, [num_words, num_filters * len(filter_sizes)]

    """
    num_words = shape(inputs, 0)
    num_chars = shape(inputs, 1)
    input_size = shape(inputs, 2)
    outputs = []

    # TODO: del the tf.cast(float32) when https://github.com/tensorflow/tensorflow/pull/12943 will be done
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv_{}".format(i)):
            w = tf.get_variable("w", [filter_size, input_size, num_filters])
            b = tf.get_variable("b", [num_filters], dtype=tf.float32)
        conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID")  # [num_words, num_chars - filter_size, num_filters]
        h = tf.nn.relu(tf.nn.bias_add(conv, b))  # [num_words, num_chars - filter_size, num_filters]
        pooled = tf.reduce_max(h, 1)  # [num_words, num_filters]
        outputs.append(pooled)
    return tf.concat(outputs, 1)  # [num_words, num_filters * len(filter_sizes)]


class CustomLSTMCell(tf.contrib.rnn.RNNCell):
    """Bi-LSTM"""
    def __init__(self, num_units, batch_size, dropout):
        """Initialize graph"""
        self._num_units = num_units
        self._dropout = dropout
        self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size], dtype=tf.float32), dropout)
        self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
        initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size], dtype=tf.float32)
        initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size], dtype=tf.float32)
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
            new_c = (1 - i) * c + i * tf.tanh(j)
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


class DocumentState(object):
    """
    Class that verifies the conll document and creates on its basis a new dictionary.
    """
    def __init__(self):
        self.doc_key = None
        self.text = []
        self.text_speakers = []
        self.speakers = []
        self.sentences = []
        self.clusters = collections.defaultdict(list)
        self.stacks = collections.defaultdict(list)

    def assert_empty(self):
        assert self.doc_key is None, 'self.doc_key is None'
        assert len(self.text) == 0, 'len(self.text) == 0'
        assert len(self.text_speakers) == 0, 'len(self.text_speakers)'
        assert len(self.sentences) == 0, 'len(self.sentences) == 0'
        assert len(self.speakers) == 0, 'len(self.speakers) == 0'
        assert len(self.clusters) == 0, 'len(self.clusters) == 0'
        assert len(self.stacks) == 0, 'len(self.stacks) == 0'

    def assert_finalizable(self):
        assert self.doc_key is not None, 'self.doc_key is not None finalizable'
        assert len(self.text) == 0, 'len(self.text) == 0_finalizable'
        assert len(self.text_speakers) == 0, 'len(self.text_speakers) == 0_finalizable'
        assert len(self.sentences) > 0, 'len(self.sentences) > 0_finalizable'
        assert len(self.speakers) > 0, 'len(self.speakers) > 0_finalizable'
        assert all(len(s) == 0 for s in self.stacks.values()), 'all(len(s) == 0 for s in self.stacks.values())_finalizable'

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
                # print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = flatten(merged_clusters)
        # In folder test one file have repeting mentions, it call below assert, everething else works fine

        return {"doc_key": self.doc_key,
                "sentences": self.sentences,
                "speakers": self.speakers,
                "clusters": merged_clusters}


def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def handle_line(line, document_state):
    """
    Updates the state of the document_state class in a read string of conll document.
    Args:
        line: line from conll document
        document_state: analyzing class

    Returns: document_state, or Nothing

    """
    if line.startswith("#begin"):
        document_state.assert_empty()
        row = line.split()
    #    document_state.doc_key = 'bc'+'{0}_{1}'.format(row[2][1:-2],row[-1])
        return None
    elif line.startswith("#end document"):
        document_state.assert_finalizable()
        return document_state.finalize()
    else:
        row = line.split('\t')
        if len(row) == 1:
            document_state.sentences.append(tuple(document_state.text))
            del document_state.text[:]
            document_state.speakers.append(tuple(document_state.text_speakers))
            del document_state.text_speakers[:]
            return None
        assert len(row) >= 12, 'len < 12'

        word = normalize_word(row[3])
        coref = row[-1]
        if document_state.doc_key == None:
            document_state.doc_key = 'bc' + '{0}_{1}'.format(row[0], row[1])

        speaker = row[8]

        word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
        document_state.text.append(word)
        document_state.text_speakers.append(speaker)

        if coref == "-":
            return None

        for segment in coref.split("|"):
            if segment[0] == "(":
                if segment[-1] == ")":
                    cluster_id = segment[1:-1]  # here was int
                    document_state.clusters[cluster_id].append((word_index, word_index))
                else:
                    cluster_id = segment[1:]  # here was int
                    document_state.stacks[cluster_id].append(word_index)
            else:
                cluster_id = segment[:-1]  # here was int
                start = document_state.stacks[cluster_id].pop()
                document_state.clusters[cluster_id].append((start, word_index))
        return None


def conll2modeldata(data):
    """
    Converts the document into a dictionary, with the required format for the model.
    Args:
        data: dict with conll string

    Returns: dict like:

    {
      "clusters": [[[1024,1024],[1024,1025]],[[876,876], [767,765], [541,544]]],
      "doc_key": "nw",
      "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
      "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
    }

    """

    conll_str = data['conll_str']
    document_state = DocumentState()
    line_list = conll_str.split('\n')
    for line in line_list:
        document = handle_line(line, document_state)
        if document is not None:
            model_file = document
    return model_file


def output_conll(input_file, predictions):
    """
    Gets the string with conll file, and write there new coreference clusters from predictions.

    Args:
        input_file: dict with conll string
        predictions: dict new clusters

    Returns: modified string with conll file

    """
    prediction_map = {}
    input_file = input_file['conll_str']
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
    new_conll = ''
    for line in input_file.split('\n'):
        if line.startswith("#begin"):
            new_conll += line + '\n'
            continue
        elif line.startswith("#end document"):
            new_conll += line
            continue
        else:
            row = line.split()
            if len(row) == 0:
                new_conll += '\n'
                continue
            
            glen = 0
            for l in row[:-1]:
                glen += len(l)
            glen += len(row[:-1])
            
            doc_key = 'bc' + '{}_{}'.format(row[0], row[1])
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
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            word_index += 1
            
            line = line[:glen] + row[-1]
            new_conll += line + '\n'
    
    return new_conll
