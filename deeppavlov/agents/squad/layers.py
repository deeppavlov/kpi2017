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

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import TimeDistributed, Lambda, Dense, multiply, LSTM, Bidirectional, Dropout

'''
----------- Lambda functions --------------
'''

def repeat_vector(vector, input):
    """Repeat vector n times along specified axis(1)."""

    vector = tf.expand_dims(vector, axis=1)
    n = tf.shape(input)[1]
    return tf.tile(vector, tf.stack([1, n, 1]))

def concatenate(values):
    """Concatenate list of tensor along specified axis(2)."""

    return tf.concat(values=values, axis=2)

def flatten(value):
    """Flatten tensor."""

    return tf.contrib.layers.flatten(value)

def masked_softmax(tensor, mask, expand=2, axis=1):
    """Masked soft-max using Lambda and merge-multiplication.

    Args:
        tensor: tensor containing scores
        mask: mask for tensor where 1 - means values at this position and 0 - means void, padded, etc..
        expand: axis along which to repeat mask
        axis: axis along which to compute soft-max

    Returns:
        masked soft-max values
    """

    mask = tf.expand_dims(mask, axis=expand)
    exponentiate = Lambda(lambda x: K.exp(x - K.max(x, axis=axis, keepdims=True)))(tensor)
    masked = tf.multiply(exponentiate, mask)
    div = tf.expand_dims(tf.reduce_sum(masked, axis=axis), axis=axis)
    predicted = tf.divide(masked, div)
    return predicted

def masked_tensor(tensor, mask):
    """Returns result of a multiplication of tensor and mask."""

    mask = tf.expand_dims(mask, axis=2)
    return tf.multiply(tensor, mask)


'''
---------- Layers -------------------------
'''

def learnable_wiq(context, question, question_mask, layer_dim):
    """Aligned question embedding. Same as in DRQA paper."""

    question_enc = TimeDistributed(Dense(units=layer_dim, activation='relu'))(question)
    context_enc = TimeDistributed(Dense(units=layer_dim, activation='relu'))(context)
    question_enc = Lambda(lambda q: tf.transpose(q, [0, 2, 1]))(question_enc)
    matrix = Lambda(lambda q: tf.matmul(q[0], q[1]))([context_enc, question_enc])
    coefs = Lambda(lambda q: masked_softmax(matrix, question_mask, axis=2, expand=1))([matrix, question_mask])
    aligned_question_enc = Lambda(lambda q: tf.matmul(q[0], q[1]))([coefs, question])
    return(aligned_question_enc)


def biLSTM_encoder(input, units, dropout, recurrent_dropout, num_layers):
    """Question and context encoder. Just Bi-LSTM from keras."""

    encoder = input
    for i in range(num_layers):
        encoder = Bidirectional(LSTM(units=units,
                                activation='tanh',
                                recurrent_activation='hard_sigmoid',
                                use_bias=True,
                                kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal',
                                bias_initializer='zeros',
                                unit_forget_bias=True,
                                kernel_regularizer=None,
                                recurrent_regularizer=None,
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                recurrent_constraint=None,
                                bias_constraint=None,
                                return_sequences=True,
                                dropout=dropout,
                                recurrent_dropout = recurrent_dropout,
                                unroll=False)) (encoder)

    return encoder

def biLSTM_encoder2(input, units, dropout = 0.0, recurrent_dropout = 0.0, num_layers = 3, input_dropout = 0.3, output_dropout = 0.3, concat_layers = True):
    """Question and context encoder. Just Bi-LSTM from keras.

    Added optional dropout between layers.
    Added optional concatenation of each layer outputs into one output representation."""

    outputs = [input]

    for i in range(num_layers):
        rnn_input = outputs[-1]

        if input_dropout > 0:
            rnn_input = Dropout(rate=input_dropout)(rnn_input)

        rnn_output = Bidirectional(LSTM(units=units,
                                activation='tanh',
                                recurrent_activation='hard_sigmoid',
                                use_bias=True,
                                kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal',
                                bias_initializer='zeros',
                                unit_forget_bias=True,
                                kernel_regularizer=None,
                                recurrent_regularizer=None,
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                recurrent_constraint=None,
                                bias_constraint=None,
                                return_sequences=True,
                                dropout=dropout,
                                recurrent_dropout = recurrent_dropout,
                                unroll=False)) (rnn_input)

        outputs.append(rnn_output)

    # Concat hidden layers
    if concat_layers:
        output = concatenate(outputs[1:])
    else:
        output = outputs[-1]

    if output_dropout > 0:
        output = Dropout(rate=input_dropout)(output)

    return output


def projection(encoding, W, dropout_rate):
    """Projection layer. Dense layer from keras.

    In FastQA is applied after the encoder, to project context and question representations
    into different spaces."""

    proj = TimeDistributed(
        Dense(W,
              trainable=True,
              weights=np.concatenate((np.eye(W), np.eye(W)), axis=1)))(encoding)
    proj = Dropout(rate=dropout_rate)(proj)
    return proj


def question_attn_vector(question_encoding, question_mask, context_encoding, repeat=True):
    """Attention over question."""

    question_attention_vector = TimeDistributed(Dense(1))(question_encoding)
    # apply masking
    question_attention_vector = Lambda(lambda q: masked_softmax(q[0], q[1]))([question_attention_vector, question_mask])
    # apply the attention
    question_attention_vector = Lambda(lambda q: q[0] * q[1])([question_encoding, question_attention_vector])
    question_attention_vector = Lambda(lambda q: K.sum(q, axis=1))(question_attention_vector)
    if repeat==True:
        question_attention_vector = Lambda(lambda q: repeat_vector(q[0], q[1]))([question_attention_vector, context_encoding])
    return question_attention_vector


def bilinear_attn(context_encoding, question_attention_vector, context_mask):
    """DRQA variant of answer start and end pointer layer. Unstable!"""

    x = context_encoding
    Wy = Lambda( lambda q: Dense(768, weights=[np.eye(768)])(q[:,0,:]))(question_attention_vector)
    xWy = Lambda(lambda q: tf.reduce_sum(tf.multiply(q[0],tf.expand_dims(q[1], 1)), axis=2, keep_dims=True))([x, Wy])

    # apply masking
    answer_start = Lambda(lambda q: masked_softmax(q[0], q[1]))([xWy, context_mask])
    answer_start = Lambda(lambda q: flatten(q))(answer_start)

    return  answer_start


def answer_start_pred(context_encoding, question_attention_vector, context_mask, W, dropout_rate):
    """Answer start prediction layer."""

    answer_start = Lambda(lambda arg:
                          concatenate([arg[0], arg[1], arg[2]]))([
        context_encoding,
        question_attention_vector,
        multiply([context_encoding, question_attention_vector])])

    answer_start = TimeDistributed(Dense(W, activation='relu'))(answer_start)
    answer_start = Dropout(rate=dropout_rate)(answer_start)
    answer_start = TimeDistributed(Dense(1))(answer_start)

    # apply masking
    answer_start = Lambda(lambda q: masked_softmax(q[0], q[1]))([answer_start, context_mask])
    answer_start = Lambda(lambda q: flatten(q))(answer_start)
    return answer_start


def answer_end_pred(context_encoding, question_attention_vector, context_mask, answer_start_distribution, W, dropout_rate):
    """Answer end prediction layer."""

    # Answer end prediction depends on the start prediction
    def s_answer_feature(x):
        maxind = K.argmax(
            x,
            axis=1,
        )
        return maxind

    x = Lambda(lambda x: K.tf.cast(s_answer_feature(x), dtype=K.tf.int32))(answer_start_distribution)
    start_feature = Lambda(lambda arg: K.tf.gather_nd(arg[0], K.tf.stack(
        [tf.range(K.tf.shape(arg[1])[0]), tf.cast(arg[1], K.tf.int32)], axis=1)))([context_encoding, x])

    start_feature = Lambda(lambda q: repeat_vector(q[0], q[1]))([start_feature, context_encoding])

    # Answer end prediction
    answer_end = Lambda(lambda arg: concatenate([
        arg[0],
        arg[1],
        arg[2],
        multiply([arg[0], arg[1]]),
        multiply([arg[0], arg[2]])
    ]))([context_encoding, question_attention_vector, start_feature])

    answer_end = TimeDistributed(Dense(W, activation='relu'))(answer_end)
    answer_end = Dropout(rate=dropout_rate)(answer_end)
    answer_end = TimeDistributed(Dense(1))(answer_end)

    # apply masking
    answer_end = Lambda(lambda q: masked_softmax(q[0], q[1]))([answer_end, context_mask])
    answer_end = Lambda(lambda q: flatten(q))(answer_end)
    return answer_end
