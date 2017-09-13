import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import TimeDistributed, Lambda, Dense, Activation, multiply, LSTM, Bidirectional, Dropout, merge
from keras.activations import softmax as Softmax
import scipy.stats as stats
from keras import backend as K
from keras.engine.topology import Layer


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

def masked_softmax(tensor, mask, expand=2, axis=1):
    '''Masked soft-max using Lambda and merge-multiplication'''
    mask = tf.expand_dims(mask, axis=expand)
    exponentiate = Lambda(lambda x: K.exp(x))(tensor)
    masked = tf.multiply(exponentiate, mask)
    div = tf.expand_dims(tf.reduce_sum(masked, axis=axis), axis=axis)
    predicted = tf.divide(masked, div)
    return predicted

def masked_tensor(tensor, mask):
    mask = tf.expand_dims(mask, axis=2)
    return tf.multiply(tensor, mask)


'''
---------- Layers -------------------------
'''

def learnable_wiq(context, question, question_mask, layer_dim):
    ''' Aligned question embedding'''
    question_enc = TimeDistributed(Dense(units=layer_dim, activation='relu'))(question)
    context_enc = TimeDistributed(Dense(units=layer_dim, activation='relu'))(context)
    question_enc = Lambda(lambda q: tf.transpose(q, [0, 2, 1]))(question_enc)
    matrix = Lambda(lambda q: tf.matmul(q[0], q[1]))([context_enc, question_enc])
    coefs = Lambda(lambda q: masked_softmax(matrix, question_mask, axis=2, expand=1))([matrix, question_mask])
    aligned_question_enc = Lambda(lambda q: tf.matmul(q[0], q[1]))([coefs, question])
    return(aligned_question_enc)



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

def biLSTM_encoder2(input, units, dropout = 0.0, recurrent_dropout = 0.0, num_layers = 3, input_dropout = 0.3, output_dropout = 0.3, concat_layers = True):
    ''' Question and context encoder '''
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
    ''' Projection layear
    in FastQA is applied after encoder, to project context and question representations
    into different spaces '''
    proj = TimeDistributed(
        Dense(W,
              use_bias=False,
              trainable=True,
              weights=np.concatenate((np.eye(W), np.eye(W)), axis=1)))(encoding)
    proj = Dropout(rate=dropout_rate)(proj)
    return proj


def question_attn_vector(question_encoding, question_mask, context_encoding, repeat=True):
    ''' Attention over question '''
    question_attention_vector = TimeDistributed(Dense(1))(question_encoding)
    # apply masking
    question_attention_vector = Lambda(lambda q: masked_softmax(q[0], q[1]))([question_attention_vector, question_mask])
    # apply the attention
    question_attention_vector = Lambda(lambda q: q[0] * q[1])([question_encoding, question_attention_vector])
    question_attention_vector = Lambda(lambda q: K.sum(q, axis=1))(question_attention_vector)
    if repeat==True:
        question_attention_vector = Lambda(lambda q: repeat_vector(q[0], q[1]))([question_attention_vector, context_encoding])
    return question_attention_vector


class BilinearProductLayer(Layer):
  def __init__(self, output_dim, input_dim=None, **kwargs):
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(BilinearProductLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    mean = 0.0
    std = 1.0
    d = self.input_dim
    initial_W_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(d,d))
    self.W = K.variable(initial_W_values)
    self.trainable_weights = [self.W]

  def call(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) <= 1:
      raise Exception('BilinearProductLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = K.shape(e1)[0]
    return K.sum((e2 * K.dot(e1, self.W[0])), axis=1)

  def compute_output_shape(self, input_shape):
    # print (input_shape)
    batch_size = input_shape[0][0]
    return (batch_size, self.output_dim)


def bilinear_attn(context_encoding, question_attention_vector, context_mask):
    ''' DRQA variant of answer start and end pointer layer '''
    xWy = TimeDistributed(BilinearProductLayer)(question_attention_vector, context_encoding)

    # apply masking
    answer_start = Lambda(lambda q: masked_softmax(q[0], q[1]))([xWy, context_mask])
    answer_start = Lambda(lambda q: flatten(q))(answer_start)

    return  answer_start #Lambda(lambda q: K.in_train_phase(lambda: tf.log(q), lambda: q))(answer_start)


def answer_start_pred(context_encoding, question_attention_vector, context_mask, W, dropout_rate):
    ''' Answer start prediction layer '''
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
    answer_end = Dropout(rate=dropout_rate)(answer_end)
    answer_end = TimeDistributed(Dense(1))(answer_end)

    # apply masking
    answer_end = Lambda(lambda q: masked_softmax(q[0], q[1]))([answer_end, context_mask])
    answer_end = Lambda(lambda q: flatten(q))(answer_end)
    return answer_end
