import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import TimeDistributed, Lambda, Dense, Activation, multiply, LSTM, Bidirectional
from keras.activations import softmax as Softmax

'''
----------- Lambda functions --------------
'''


def repeat_vector(vector, input):
    ''' Repeats vector n times along specified axis '''
    vector = tf.expand_dims(vector, axis=1)
    n = tf.shape(input)[1]
    return tf.tile(vector, tf.stack([1, n, 1]))

def concatenate(values):
    return tf.concat(values=values, axis=2)

def flatten(value):
    return tf.contrib.layers.flatten(value)

'''
---------- Layers -------------------------
'''

def learnable_wiq(question, context, layer_dim):
    ''' '''
    question_enc = TimeDistributed(Dense(units=layer_dim, activation='relu'))(question)
    context_enc = TimeDistributed(Dense(units=layer_dim, activation='relu'))(context)


def biLSTM_encoder(input, units, dropout, recurrent_dropout, num_layers):
    ''' Question and context encoder '''
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


def projection(encoding, W):
    return TimeDistributed(
        Dense(W,
              use_bias=False,
              trainable=True,
              weights=np.concatenate((np.eye(W), np.eye(W)), axis=1)))(encoding)


def question_attn_vector(question_encoding, context_encoding):
    ''' Attention over question '''
    question_attention_vector = TimeDistributed(Dense(1))(question_encoding)
    question_attention_vector = Lambda(lambda q: Softmax(q, axis=1))(question_attention_vector)
    # apply the attention
    question_attention_vector = Lambda(lambda q: q[0] * q[1])([question_encoding, question_attention_vector])
    question_attention_vector = Lambda(lambda q: K.sum(q, axis=1))(question_attention_vector)
    question_attention_vector = Lambda(lambda q: repeat_vector(q[0], q[1]))([question_attention_vector, context_encoding])
    question_attention_vector = Lambda(lambda q: Softmax(q, axis=1))(question_attention_vector)
    return question_attention_vector


def answer_start_pred(context_encoding, question_attention_vector, W):
    ''' Answer start prediction layer '''
    answer_start = Lambda(lambda arg:
                          concatenate([arg[0], arg[1], arg[2]]))([
        context_encoding,
        question_attention_vector,
        multiply([context_encoding, question_attention_vector])])

    answer_start = TimeDistributed(Dense(W, activation='relu'))(answer_start)
    answer_start = TimeDistributed(Dense(1))(answer_start)
    answer_start = Lambda(lambda q: flatten(q))(answer_start)
    answer_start = Activation('softmax')(answer_start)
    return answer_start


def answer_end_pred(context_encoding, question_attention_vector, answer_start_distribution, W):
    ''' Answer end prediction layer '''

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
    answer_end = TimeDistributed(Dense(1))(answer_end)
    answer_end = Lambda(lambda q: flatten(q))(answer_end)
    answer_end = Activation('softmax')(answer_end)
    return answer_end
