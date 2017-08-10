from .metrics import fbeta_score

import os
import numpy as np
import copy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from keras.layers import Dense, Activation, Input, Embedding, LSTM, Dropout, multiply, Lambda
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.initializers import glorot_uniform, Orthogonal
from keras import backend as K
from keras.optimizers import Adam


class ParaphraserModel(object):

    def __init__(self, word_index, embedding_matrix, opt):
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
        self.opt = copy.deepcopy(opt)
        self.max_sequence_length = opt['max_sequence_length']
        self.embedding_dim = opt['embedding_dim']
        self.learning_rate = opt['learning_rate']
        self.batch_size = opt['batch_size']
        self.epoch_num = opt['epoch_num']
        self.seed = opt['seed']
        self.hidden_dim = opt['hidden_dim']
        self.attention_dim = opt['attention_dim']
        self.perspective_num = opt['perspective_num']
        self.aggregation_dim = opt['aggregation_dim']
        self.dense_dim = opt['dense_dim']
        self.ldrop_val = opt['ldrop_val']
        self.dropout_val = opt['dropout_val']
        self.recdrop_val = opt['recdrop_val']
        self.inpdrop_val = opt['inpdrop_val']
        self.ldropagg_val = opt['ldropagg_val']
        self.dropoutagg_val = opt['dropoutagg_val']
        self.recdropagg_val = opt['recdropagg_val']
        self.inpdropagg_val = opt['inpdropagg_val']
        self.model_name = opt['model_name']

        if self.opt.get('model_file') and os.path.isfile(opt['model_file']):
            self._init_from_saved(opt['model_file'])
        else:
            if self.opt.get('pretrained_model'):
                self._init_from_saved(opt['pretrained_model'])
            else:
                self._init_from_scratch()
        self.opt['cuda'] = not self.opt['no_cuda']
        if self.opt['cuda']:
            print('[ Using CUDA (GPU %d) ]' % opt['gpu'])
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.45
            config.gpu_options.visible_device_list = str(opt['gpu'])
            set_session(tf.Session(config=config))

        self.n_examples = 0
        self.updates = 0

    def _init_from_scratch(self):
        print('[ Initializing model from scratch ]')
        if self.model_name == 'bmwacor':
            self.model = self.bmwacor_model()
        if self.model_name == 'bmwacor1':
            self.model = self.bmwacor1_model()
        if self.model_name == 'bilstm_split':
            self.model = self.bilstm_split_model()
        if self.model_name == 'bilstm_split1':
            self.model = self.bilstm_split1_model()
        if self.model_name == 'full_match':
            self.model = self.full_match_model()
        if self.model_name == 'maxpool_match':
            self.model = self.maxpool_match_model()
        if self.model_name == 'att_match':
            self.model = self.att_match_model()
        if self.model_name == 'maxatt_match':
            self.model = self.maxatt_match_model()
        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy', fbeta_score])

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save(fname+'.h5')

    def _init_from_saved(self, fname):
        print('[ Loading model %s ]' % fname)
        self.model = load_model(fname)

    def update(self, batch):
        x, y = batch
        self.model.train_on_batch(x, y)
        self.updates += 1

    def predict(self, batch):
        self.model.predict_on_batch(batch)

    def create_embedding_layer(self, input_dim):
        inp = Input(shape=(input_dim,))
        out = Embedding(len(self.word_index) + 1,
                        self.embedding_dim,
                        weights=[self.embedding_matrix],
                        input_length=input_dim,
                        trainable=False)(inp)
        model = Model(inputs=inp, outputs=out, name="word_embedding_model")
        return model

    def create_lstm_layer(self, input_dim):
        inp = Input(shape=(input_dim, self.embedding_dim,))
        inp_dropout = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        outp = LSTM(self.hidden_dim, input_shape=(input_dim, self.embedding_dim,),
                    kernel_regularizer=None,
                    recurrent_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    recurrent_dropout=self.recdrop_val,
                    dropout=self.inpdrop_val,
                    kernel_initializer=ker_in,
                    recurrent_initializer=rec_in,
                    return_sequences=True)(inp_dropout)
        outp_dropout = Dropout(self.dropout_val)(outp)
        model = Model(inputs=inp, outputs=outp_dropout, name="LSTM_encoder")
        return model

    def create_lstm_layer_1(self, input_dim):
        inp = Input(shape=(input_dim,  self.embedding_dim,))
        inp_drop = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(input_dim, self.embedding_dim,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=True), merge_mode=None)(inp_drop)
        dropout_forw = Dropout(self.dropout_val)(bioutp[0])
        dropout_back = Dropout(self.dropout_val)(bioutp[1])
        model = Model(inputs=inp, outputs=[dropout_forw, dropout_back], name="biLSTM_encoder")
        return model

    def create_lstm_layer_2(self, input_dim):
        inp = Input(shape=(input_dim, 2*self.perspective_num,))
        inp_drop = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.aggregation_dim,
                                    input_shape=(input_dim, 2*self.perspective_num,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=True), merge_mode=None)(inp_drop)
        dropout_forw = Dropout(self.dropout_val)(bioutp[0])
        dropout_back = Dropout(self.dropout_val)(bioutp[1])
        model = Model(inputs=inp, outputs=[dropout_forw, dropout_back], name="biLSTM_enc_persp")
        return model

    def create_attention_layer(self, input_dim_a, input_dim_b):
        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        val = np.concatenate((np.zeros((self.max_sequence_length-1,1)), np.ones((1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp_b_perm)
        ker_in = glorot_uniform(seed=self.seed)
        outp_a = Dense(self.attention_dim, input_shape=(input_dim_a, self.hidden_dim),
                       kernel_initializer=ker_in, activation='relu')(inp_a)
        outp_last = Dense(self.attention_dim, input_shape=(1, self.hidden_dim),
                          kernel_initializer=ker_in, activation='relu')(last_state)
        outp_last_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_last)
        outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last_perm, outp_a])
        outp_norm = Activation('softmax')(outp)
        outp_norm_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_norm)
        model = Model(inputs=[inp_a, inp_b], outputs=outp_norm_perm, name="attention_generator")
        return model

    def create_attention_layer_f(self, input_dim_a, input_dim_b):
        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        val = np.concatenate((np.zeros((self.max_sequence_length-1,1)), np.ones((1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp_b_perm)
        ker_in = glorot_uniform(seed=self.seed)
        outp_a = Dense(self.attention_dim, input_shape=(input_dim_a, self.hidden_dim),
                       kernel_initializer=ker_in, activation='relu')(inp_a)
        outp_last = Dense(self.attention_dim, input_shape=(1, self.hidden_dim),
                          kernel_initializer=ker_in, activation='relu')(last_state)
        outp_last_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_last)
        outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last_perm, outp_a])
        outp_norm = Activation('softmax')(outp)
        outp_norm_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_norm)
        model = Model(inputs=[inp_a, inp_b], outputs=outp_norm_perm, name="att_generator_forw")
        return model

    def create_attention_layer_b(self, input_dim_a, input_dim_b):
        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        val = np.concatenate((np.ones((1,1)), np.zeros((self.max_sequence_length-1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp_b_perm)
        ker_in = glorot_uniform(seed=self.seed)
        outp_a = Dense(self.attention_dim, input_shape=(input_dim_a, self.hidden_dim),
                       kernel_initializer=ker_in, activation='relu')(inp_a)
        outp_last = Dense(self.attention_dim, input_shape=(1, self.hidden_dim),
                          kernel_initializer=ker_in, activation='relu')(last_state)
        outp_last_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_last)
        outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last_perm, outp_a])
        outp_norm = Activation('softmax')(outp)
        outp_norm_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_norm)
        model = Model(inputs=[inp_a, inp_b], outputs=outp_norm_perm, name="att_generator_back")
        return model

    def weighted_with_attention(self, inputs):
        inp, inp_cont = inputs
        val = np.eye(self.max_sequence_length)
        kcon = K.constant(value=val, dtype='float32')
        diag = K.repeat_elements(inp_cont, self.max_sequence_length, 2) * kcon
        return K.batch_dot(diag, K.permute_dimensions(inp, (0,2,1)), axes=[1,2])

    def weighted_with_attention_output_shape(self, shapes):
        shape1, shape2 = shapes
        return shape1

    def dim_reduction(self, inp):
        return K.sum(inp, axis=1)

    def dim_reduction_output_shape(self, shape):
        return shape[0], shape[2]

    def weight_and_reduce(self, inputs):
        inp, inp_cont = inputs
        reduced = K.batch_dot(inp_cont,
                              K.permute_dimensions(inp, (0,2,1)), axes=[1,2])
        return K.squeeze(reduced, 1)

    def weight_and_reduce_output_shape(self, shapes):
        shape1, shape2 = shapes
        return shape1[0], shape1[2]

    def cosine_dist(self, inputs):
        input1, input2 = inputs
        a = K.abs(input1-input2)
        b = multiply(inputs)
        return K.concatenate([a, b])

    def cosine_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return shape1[0], 2*shape1[1]

    def create_full_matching_layer_f(self, input_dim_a, input_dim_b):
        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        W = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0, seed=self.seed)
            W.append(wi)

        val = np.concatenate((np.zeros((self.max_sequence_length-1,1)), np.ones((1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp_b_perm)
        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * W[i])(inp_a)
            outp_last = Lambda(lambda x: x * W[i])(last_state)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_last = Lambda(lambda x: K.l2_normalize(x, -1))(outp_last)
            outp_last = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_last)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last, outp_a])
            outp = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def create_full_matching_layer_b(self, input_dim_a, input_dim_b):
        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        W = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0, seed=self.seed)
            W.append(wi)

        val = np.concatenate((np.ones((1, 1)), np.zeros((self.max_sequence_length - 1, 1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0, 2, 1)))(inp_b_perm)
        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * W[i])(inp_a)
            outp_last = Lambda(lambda x: x * W[i])(last_state)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_last = Lambda(lambda x: K.l2_normalize(x, -1))(outp_last)
            outp_last = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_last)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last, outp_a])
            outp = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def create_maxpool_matching_layer(self, input_dim_a, input_dim_b):
        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        W = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0, seed=self.seed)
            W.append(wi)

        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * W[i])(inp_a)
            outp_b = Lambda(lambda x: x * W[i])(inp_b)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_b = Lambda(lambda x: K.l2_normalize(x, -1))(outp_b)
            outp_b = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_b)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_b, outp_a])
            outp = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp)
            outp = Lambda(lambda x: K.max(x, -1, keepdims=True))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def create_att_matching_layer(self, input_dim_a, input_dim_b):
        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))

        w = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0, seed=self.seed)
            w.append(wi)

        outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(inp_a)
        outp_b = Lambda(lambda x: K.l2_normalize(x, -1))(inp_b)
        outp_b = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_b)
        alpha = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_b, outp_a])
        alpha = Lambda(lambda x: K.l2_normalize(x, 1))(alpha)
        hmean = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([alpha, outp_b])

        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * w[i])(inp_a)
            outp_hmean = Lambda(lambda x: x * w[i])(hmean)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_hmean = Lambda(lambda x: K.l2_normalize(x, -1))(outp_hmean)
            outp_hmean = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_hmean)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_hmean, outp_a])
            val = np.eye(self.max_sequence_length)
            kcon = K.constant(value=val, dtype='float32')
            outp = Lambda(lambda x: K.sum(x * kcon, -1, keepdims=True))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def create_maxatt_matching_layer(self, input_dim_a, input_dim_b):
        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))

        W = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0, seed=self.seed)
            W.append(wi)

        outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(inp_a)
        outp_b = Lambda(lambda x: K.l2_normalize(x, -1))(inp_b)
        outp_b = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_b)
        alpha = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_b, outp_a])
        alpha = Lambda(lambda x: K.one_hot(K.argmax(x, 1), self.max_sequence_length))(alpha)
        hmax = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([alpha, outp_b])

        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * W[i])(inp_a)
            outp_hmax = Lambda(lambda x: x * W[i])(hmax)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_hmax = Lambda(lambda x: K.l2_normalize(x, -1))(outp_hmax)
            outp_hmax = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_hmax)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_hmax, outp_a])
            val = np.eye(self.max_sequence_length)
            kcon = K.constant(value=val, dtype='float32')
            outp = Lambda(lambda x: K.sum(x * kcon, -1, keepdims=True))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def cosine_dist(self, inputs):
        input1, input2 = inputs
        a = K.abs(input1-input2)
        b = multiply(inputs)
        return K.concatenate([a, b])

    def cosine_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return shape1[0], 2*shape1[1]


    def terminal_f(self, inp):
        val = np.concatenate((np.zeros((self.max_sequence_length-1,1)), np.ones((1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp)
        return K.squeeze(last_state, 1)

    def terminal_f_output_shape(self, shape):
        return shape[0], shape[2]

    def terminal_b(self, inp):
        val = np.concatenate((np.ones((1,1)), np.zeros((self.max_sequence_length-1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp)
        return K.squeeze(last_state, 1)

    def terminal_b_output_shape(self, shape):
        return shape[0], shape[2]

    def bmwacor_model(self):
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        embedding_layer = self.create_embedding_layer(self.max_sequence_length)
        lstm_layer = self.create_lstm_layer(self.max_sequence_length)
        attention_layer = self.create_attention_layer(self.max_sequence_length, self.max_sequence_length)
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        lstm_a = lstm_layer(embedding_a)
        lstm_b = lstm_layer(embedding_b)
        attention_a = attention_layer([lstm_a, lstm_b])
        attention_b = attention_layer([lstm_b, lstm_a])

        reduced_a = Lambda(self.weight_and_reduce,
                           output_shape=self.weight_and_reduce_output_shape, name="mul_sum_q1")([lstm_a, attention_a])
        reduced_b = Lambda(self.weight_and_reduce,
                           output_shape=self.weight_and_reduce_output_shape, name="mul_sum_q2")([lstm_b, attention_b])

        dist = Lambda(self.cosine_dist, output_shape=self.cosine_dist_output_shape,
                      name="similarity_network")([reduced_a, reduced_b])

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dist)

        model = Model([input_a, input_b], dense)

        return model

    def bmwacor1_model(self):
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        embedding_layer = self.create_embedding_layer(self.max_sequence_length)
        lstm_layer = self.create_lstm_layer(self.max_sequence_length)
        attention_layer = self.create_attention_layer(self.max_sequence_length, self.max_sequence_length)
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        lstm_a = lstm_layer(embedding_a)
        lstm_b = lstm_layer(embedding_b)
        attention_a = attention_layer([lstm_a, lstm_b])
        attention_b = attention_layer([lstm_b, lstm_a])

        weighted_a = Lambda(self.weighted_with_attention,
                            output_shape=self.weighted_with_attention_output_shape, name="mul_q1")([lstm_a, attention_a])
        weighted_b = Lambda(self.weighted_with_attention,
                            output_shape=self.weighted_with_attention_output_shape, name="mul_q2")([lstm_b, attention_b])

        reduced_a = Lambda(self.dim_reduction,
                           output_shape=self.dim_reduction_output_shape, name="sum_q1")(weighted_a)
        reduced_b = Lambda(self.dim_reduction,
                           output_shape=self.dim_reduction_output_shape, name="sum_q2")(weighted_b)

        dist = Lambda(self.cosine_dist, output_shape=self.cosine_dist_output_shape,
                      name="similarity_network")([reduced_a, reduced_b])

        ker_in = glorot_uniform(seed=None)
        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_initializer=ker_in)(dist)

        model = Model([input_a, input_b], dense)

        return model

    def bilstm_split_model(self):
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        embedding_layer = self.create_embedding_layer(self.max_sequence_length)
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        attention_layer_f = self.create_attention_layer_f(self.max_sequence_length, self.max_sequence_length)
        attention_layer_b = self.create_attention_layer_b(self.max_sequence_length, self.max_sequence_length)
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        lstm_a = lstm_layer(embedding_a)
        lstm_b = lstm_layer(embedding_b)
        attention_a_forw = attention_layer_f([lstm_a[0], lstm_b[0]])
        attention_a_back = attention_layer_b([lstm_a[1], lstm_b[1]])
        attention_b_forw = attention_layer_f([lstm_b[0], lstm_a[0]])
        attention_b_back = attention_layer_b([lstm_b[1], lstm_a[1]])

        reduced_a_forw = Lambda(self.weight_and_reduce,
                                output_shape=self.weight_and_reduce_output_shape,
                                name="sum_q1_f")([lstm_a[0], attention_a_forw])
        reduced_a_back = Lambda(self.weight_and_reduce,
                                output_shape=self.weight_and_reduce_output_shape,
                                name="sum_q1_b")([lstm_a[1], attention_a_back])
        reduced_b_forw = Lambda(self.weight_and_reduce,
                                output_shape=self.weight_and_reduce_output_shape,
                                name="sum_q2_f")([lstm_b[0], attention_b_forw])
        reduced_b_back = Lambda(self.weight_and_reduce,
                                output_shape=self.weight_and_reduce_output_shape,
                                name="sum_q2_b")([lstm_b[1], attention_b_back])

        reduced_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                           name='concat_q1')([reduced_a_forw, reduced_a_back])
        reduced_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                           name='concat_q2')([reduced_b_forw, reduced_b_back])

        dist = Lambda(self.cosine_dist, output_shape=self.cosine_dist_output_shape,
                      name="similarity_network")([reduced_a, reduced_b])

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dist)

        model = Model([input_a, input_b], dense)

        return model

    def bilstm_split1_model(self):
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        embedding_layer = self.create_embedding_layer(self.max_sequence_length)
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        attention_layer_f = self.create_attention_layer_f(self.max_sequence_length, self.max_sequence_length)
        attention_layer_b = self.create_attention_layer_b(self.max_sequence_length, self.max_sequence_length)
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        lstm_a = lstm_layer(embedding_a)
        lstm_b = lstm_layer(embedding_b)
        attention_a_forw = attention_layer_f([lstm_a[0], lstm_b[0]])
        attention_a_back = attention_layer_b([lstm_a[1], lstm_b[1]])
        attention_b_forw = attention_layer_f([lstm_b[0], lstm_a[0]])
        attention_b_back = attention_layer_b([lstm_b[1], lstm_a[1]])

        weighted_a_forw = Lambda(self.weighted_with_attention,
                                 output_shape=self.weighted_with_attention_output_shape,
                                 name="mul_q1_f")([lstm_a[0], attention_a_forw])
        weighted_a_back = Lambda(self.weighted_with_attention,
                                 output_shape=self.weighted_with_attention_output_shape,
                                 name="mul_q1_b")([lstm_a[1], attention_a_back])
        weighted_b_forw = Lambda(self.weighted_with_attention,
                                 output_shape=self.weighted_with_attention_output_shape,
                                 name="mul_q2_f")([lstm_b[0], attention_b_forw])
        weighted_b_back = Lambda(self.weighted_with_attention,
                                 output_shape=self.weighted_with_attention_output_shape,
                                 name="mul_q2_b")([lstm_b[1], attention_b_back])

        reduced_a_forw = Lambda(self.dim_reduction,
                                output_shape=self.dim_reduction_output_shape, name="sum_q1_f")(weighted_a_forw)
        reduced_a_back = Lambda(self.dim_reduction,
                                output_shape=self.dim_reduction_output_shape, name="sum_q1_b")(weighted_a_back)
        reduced_b_forw = Lambda(self.dim_reduction,
                                output_shape=self.dim_reduction_output_shape, name="sum_q2_f")(weighted_b_forw)
        reduced_b_back = Lambda(self.dim_reduction,
                                output_shape=self.dim_reduction_output_shape, name="sum_q2_b")(weighted_b_back)

        reduced_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                           name='concat_q1')([reduced_a_forw, reduced_a_back])
        reduced_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                           name='concat_q2')([reduced_b_forw, reduced_b_back])

        dist = Lambda(self.cosine_dist, output_shape=self.cosine_dist_output_shape,
                      name="similarity_network")([reduced_a, reduced_b])

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dist)

        model = Model([input_a, input_b], dense)

        return model

    def maxpool_match_model(self):
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        embedding_layer = self.create_embedding_layer(self.max_sequence_length)
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        matching_layer_f = self.create_maxpool_matching_layer(self.max_sequence_length, self.max_sequence_length)
        matching_layer_b = self.create_maxpool_matching_layer(self.max_sequence_length, self.max_sequence_length)

        lstm_layer_agg = self.create_lstm_layer_2(self.max_sequence_length)
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        lstm_a = lstm_layer(embedding_a)
        lstm_b = lstm_layer(embedding_b)
        matching_a_forw = matching_layer_f([lstm_a[0], lstm_b[0]])
        matching_a_back = matching_layer_b([lstm_a[1], lstm_b[1]])
        matching_b_forw = matching_layer_f([lstm_b[0], lstm_a[0]])
        matching_b_back = matching_layer_b([lstm_b[1], lstm_a[1]])

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q1_match')([matching_a_forw, matching_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q2_match')([matching_b_forw, matching_b_back])

        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        reduced_a_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q1_f")(agg_a[0])
        reduced_a_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q1_b")(agg_a[1])
        reduced_b_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q2_f")(agg_b[0])
        reduced_b_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q2_b")(agg_b[1])

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1),
                         name='conc_agg')([reduced_a_forw, reduced_a_back,
                                           reduced_b_forw, reduced_b_back])

        ker_in = glorot_uniform(seed=self.seed)
        dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dense)

        model = Model([input_a, input_b], dense)
        return model

    def maxatt_match_model(self):
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        embedding_layer = self.create_embedding_layer(self.max_sequence_length)
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        matching_layer_f = self.create_maxatt_matching_layer(self.max_sequence_length, self.max_sequence_length)
        matching_layer_b = self.create_maxatt_matching_layer(self.max_sequence_length, self.max_sequence_length)

        lstm_layer_agg = self.create_lstm_layer_2(self.max_sequence_length)
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        lstm_a = lstm_layer(embedding_a)
        lstm_b = lstm_layer(embedding_b)
        matching_a_forw = matching_layer_f([lstm_a[0], lstm_b[0]])
        matching_a_back = matching_layer_b([lstm_a[1], lstm_b[1]])
        matching_b_forw = matching_layer_f([lstm_b[0], lstm_a[0]])
        matching_b_back = matching_layer_b([lstm_b[1], lstm_a[1]])

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q1_match')([matching_a_forw, matching_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q2_match')([matching_b_forw, matching_b_back])

        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        reduced_a_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q1_f")(agg_a[0])
        reduced_a_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q1_b")(agg_a[1])
        reduced_b_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q2_f")(agg_b[0])
        reduced_b_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q2_b")(agg_b[1])

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1),
                         name='conc_agg')([reduced_a_forw, reduced_a_back,
                                           reduced_b_forw, reduced_b_back])

        ker_in = glorot_uniform(seed=self.seed)
        dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dense)

        model = Model([input_a, input_b], dense)
        return model

    def att_match_model(self):
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        embedding_layer = self.create_embedding_layer(self.max_sequence_length)
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        matching_layer_f = self.create_att_matching_layer(self.max_sequence_length, self.max_sequence_length)
        matching_layer_b = self.create_att_matching_layer(self.max_sequence_length, self.max_sequence_length)

        lstm_layer_agg = self.create_lstm_layer_2(self.max_sequence_length)
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        lstm_a = lstm_layer(embedding_a)
        lstm_b = lstm_layer(embedding_b)
        matching_a_forw = matching_layer_f([lstm_a[0], lstm_b[0]])
        matching_a_back = matching_layer_b([lstm_a[1], lstm_b[1]])
        matching_b_forw = matching_layer_f([lstm_b[0], lstm_a[0]])
        matching_b_back = matching_layer_b([lstm_b[1], lstm_a[1]])

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q1_match')([matching_a_forw, matching_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q2_match')([matching_b_forw, matching_b_back])

        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        reduced_a_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q1_f")(agg_a[0])
        reduced_a_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q1_b")(agg_a[1])
        reduced_b_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q2_f")(agg_b[0])
        reduced_b_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q2_b")(agg_b[1])

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1),
                         name='conc_agg')([reduced_a_forw, reduced_a_back,
                                           reduced_b_forw, reduced_b_back])

        ker_in = glorot_uniform(seed=self.seed)
        dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dense)

        model = Model([input_a, input_b], dense)
        return model

    def full_match_model(self):
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        embedding_layer = self.create_embedding_layer(self.max_sequence_length)
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        matching_layer_f = self.create_full_matching_layer_f(self.max_sequence_length, self.max_sequence_length)
        matching_layer_b = self.create_full_matching_layer_b(self.max_sequence_length, self.max_sequence_length)

        lstm_layer_agg = self.create_lstm_layer_2(self.max_sequence_length)
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        lstm_a = lstm_layer(embedding_a)
        lstm_b = lstm_layer(embedding_b)
        matching_a_forw = matching_layer_f([lstm_a[0], lstm_b[0]])
        matching_a_back = matching_layer_b([lstm_a[1], lstm_b[1]])
        matching_b_forw = matching_layer_f([lstm_b[0], lstm_a[0]])
        matching_b_back = matching_layer_b([lstm_b[1], lstm_a[1]])

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q1_match')([matching_a_forw, matching_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q2_match')([matching_b_forw, matching_b_back])

        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        reduced_a_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q1_f")(agg_a[0])
        reduced_a_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q1_b")(agg_a[1])
        reduced_b_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q2_f")(agg_b[0])
        reduced_b_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q2_b")(agg_b[1])

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1),
                         name='conc_agg')([reduced_a_forw, reduced_a_back,
                                           reduced_b_forw, reduced_b_back])

        ker_in = glorot_uniform(seed=self.seed)
        dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dense)

        model = Model([input_a, input_b], dense)
        return model

    #def compile_model(self):
    #    optimizer = Adam(lr=self.learning_rate)
    #    model.compile(loss='binary_crossentropy',
    #                  optimizer=optimizer,
    #                  metrics=['accuracy', metrics.fbeta_score])
