# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import json, os, copy
import pickle
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda,  multiply, Masking, Dropout
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.activations import softmax as Softmax
from keras.optimizers import Adamax, Adam, Adadelta
from keras.callbacks import ModelCheckpoint
from .utils import AverageMeter, getOptimizer, score

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

# import layers
from .layers import *

'''
--------------- Model ----------------------------
'''

class SquadModel(object):
    def __init__(self, opt, word_dict = None, feature_dict = None, weights_path = None ):

        self.opt = copy.deepcopy(opt)

        for k, v in opt.items():
            setattr(self, k, v)

        self.word_dict = word_dict
        self.feature_dict = feature_dict

        self.n_examples = 0
        self.updates = 0
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.train_f1 = AverageMeter()
        self.train_em = AverageMeter()

        if self.type == 'fastqa':
            self.model = self.fastqa_default()
        elif self.type == 'drqa_clone':
            self.model = self.drqa_default()
        else:
            raise NameError('There is no model with name: {}'.format(self.model))

        optimizer = getOptimizer(self.optimizer, self.exp_decay, self.grad_norm_clip, self.lr)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        if not weights_path==None:
            print('[ Loading model %s ]' % weights_path)
            if os.path.isfile(weights_path + '.h5'):
                self.model.load_weights(weights_path + '.h5')
            else:
                print('Error. There is no %s.h5 file provided.' % weights_path)


    def save(self, fname):
        self.model.save_weights(fname+'.h5')

        params = {
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'config': self.opt,
        }

        with open(fname+'.pkl', 'wb') as f:
            pickle.dump(params, f)

    def update(self, batch):

        def cat(target):
            dtype = K.floatx()
            indices = np.arange(len(target))
            classes = batch[0].shape[1]
            batch_target = [min(target[i], classes - 1) for i in indices]
            return np_utils.to_categorical(batch_target, classes).astype(dtype)

        answ_start = []
        answ_end = []

        x, y = [batch[0], batch[1], batch[3], batch[2], batch[4]], [cat(batch[5]), cat(batch[6])]

        output = self.model.train_on_batch(x, y)
        self.train_loss.update(output[0])
        self.train_acc.update((output[3] + output[4])/2)
        self.updates += 1

        # Sometimes update F1 training score to be aware of overfitting
        if self.updates % 5 == 0:
            text = batch[-2]
            spans = batch[-1]
            answ_s, answ_e = batch[-4], batch[-3]
            answers = []
            for i in range(len(text)):
                answers.append(text[i][spans[i][answ_s[i]][0]:spans[i][answ_e[i]][1]])
            predictions = self.predict(batch)
            scorer = score(predictions, answers)
            self.train_f1.update(scorer[1])
            self.train_em.update(scorer[0])

    def predict(self, batch):

        score_s, score_e = self.model.predict_on_batch([batch[0], batch[1], batch[3], batch[2], batch[4]])

        text = batch[-2]
        spans = batch[-1]
        predictions = []

        # Return predictions in the form of text
        max_len = self.answ_maxlen or score_s.shape[1]

        for i in range(score_s.shape[0]):
            scores = np.outer(score_s[i], score_e[i])
            scores = np.tril(np.triu(scores), max_len-1)
            s_ind, e_ind = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_ind][0], spans[i][e_ind][1]
            predictions.append(text[i][s_offset:e_offset])

        return predictions


    def build_ex(self, ex):
        if 'text' not in ex:
            return

        '''Find the token span of the answer in the context for this example.
        '''
        inputs = dict()
        texts = ex['text'].split('\n')
        inputs['context'] = texts[1]
        inputs['question'] = texts[2]
        if 'labels' in ex:
            inputs['answer'] = ex['labels']

        return inputs


    def drqa_default(self):

        '''Inputs'''
        P = Input(shape=(None, self.word_embedding_dim), name='context_input')
        Q = Input(shape=(None, self.word_embedding_dim), name='question_input')
        P_f = Input(shape=(None, self.context_embedding_dim - self.word_embedding_dim), name='context_features')

        '''Masking inputs'''
        P_mask = Input(shape=(None,), name='context_mask')
        Q_mask = Input(shape=(None,), name='question_mask')

        '''Emdedding dropout (with similar mask for all timesteps)'''
        P_drop = Dropout(rate=self.embedding_dropout)(P)
        Q_drop = Dropout(rate=self.embedding_dropout)(Q)

        ''' Aligned question embedding '''
        aligned_question = learnable_wiq(P_drop, Q_drop, Q_mask, layer_dim=self.aligned_question_dim)
        passage_input = Lambda(lambda q: tf.concat(q, axis=2))([P_drop, P_f, aligned_question])

        question_input = Q_drop

        ''' Encoding '''
        passage_encoding = passage_input
        passage_encoding = Masking()(passage_encoding)
        passage_encoding = Lambda(lambda q: biLSTM_encoder2(
            q,
            self.encoder_hidden_dim,
            self.rnn_dropout,
            self.recurrent_dropout,
            self.question_enc_layers,
            self.input_dropout,
            self.output_dropout,
            True
        ))(passage_encoding)
        passage_encoding = Lambda(lambda q: masked_tensor(q[0], q[1]))([passage_encoding, P_mask])

        question_encoding = question_input
        question_encoding = Masking()(question_encoding)
        question_encoding = Lambda(lambda  q: biLSTM_encoder2(
            q,
            self.encoder_hidden_dim,
            self.rnn_dropout,
            self.recurrent_dropout,
            self.context_enc_layers,
            self.input_dropout,
            self.output_dropout,
            True
        ))(question_encoding)
        question_encoding = Lambda(lambda q: masked_tensor(q[0], q[1]))([question_encoding, Q_mask])

        '''Attention over question'''
        question_attention_vector = question_attn_vector(question_encoding, Q_mask, passage_encoding)

        '''Answer span prediction'''
        # Answer start prediction
        answer_start = bilinear_attn(passage_encoding, question_attention_vector, P_mask)
        # Answer end prediction
        answer_end = bilinear_attn(passage_encoding, question_attention_vector, P_mask)

        input_placeholders = [P, P_f, Q, P_mask, Q_mask]
        inputs = input_placeholders
        outputs = [answer_start, answer_end]

        model = Model(inputs=inputs, outputs=outputs)
        return model


    def fastqa_default(self):

        '''Inputs'''
        P = Input(shape=(None, self.word_embedding_dim), name='context_input')
        Q = Input(shape=(None, self.word_embedding_dim), name='question_input')
        P_f = Input(shape=(None, self.context_embedding_dim - self.word_embedding_dim), name='context_features')

        '''Masking inputs'''
        P_mask = Input(shape=(None,), name='context_mask')
        Q_mask = Input(shape=(None,), name='question_mask')

        passage_input = Lambda(lambda q: tf.concat(q, axis=2))([P, P_f])
        question_input = Q

        ''' Aligned question embedding '''
        aligned_question = learnable_wiq(P, Q, Q_mask, layer_dim=self.aligned_question_dim)
        passage_input = Lambda(lambda q: tf.concat(q, axis=2))([P, P_f, aligned_question])

        ''' Emdedding dropout (with similar mask for all timesteps) '''
        passage_input = Dropout(
            rate=self.embedding_dropout,
            noise_shape=(tf.shape(P)[0], 1, self.context_embedding_dim + self.aligned_question_dim))(passage_input)

        question_input = Dropout(
            rate=self.embedding_dropout,
            noise_shape=(tf.shape(Q)[0], 1, self.word_embedding_dim))(question_input)

        ''' Encoding '''
        passage_encoding = passage_input
        passage_encoding = Masking()(passage_encoding)
        passage_encoding = Lambda(lambda q: biLSTM_encoder2(
            q,
            self.encoder_hidden_dim,
            self.rnn_dropout,
            self.recurrent_dropout,
            self.question_enc_layers,
            self.input_dropout,
            self.output_dropout,
            True
        ))(passage_encoding)
        passage_encoding = Lambda(lambda q: masked_tensor(q[0], q[1]))([passage_encoding, P_mask])

        question_encoding = question_input
        question_encoding = Masking()(question_encoding)
        question_encoding = Lambda(lambda  q: biLSTM_encoder2(
            q,
            self.encoder_hidden_dim,
            self.rnn_dropout,
            self.recurrent_dropout,
            self.context_enc_layers,
            self.input_dropout,
            self.output_dropout,
            True
        ))(question_encoding)
        question_encoding = Lambda(lambda q: masked_tensor(q[0], q[1]))([question_encoding, Q_mask])

        '''Attention over question'''
        question_attention_vector = question_attn_vector(question_encoding, Q_mask, passage_encoding)

        '''Answer span prediction'''
        # Answer start prediction
        answer_start = answer_start_pred(passage_encoding, question_attention_vector, P_mask, 50, self.linear_dropout)
        # Answer end prediction
        answer_end = answer_end_pred(passage_encoding, question_attention_vector, P_mask, answer_start, 50, self.linear_dropout)

        input_placeholders = [P, P_f, Q, P_mask, Q_mask]
        inputs = input_placeholders
        outputs = [answer_start, answer_end]

        model = Model(inputs=inputs, outputs=outputs)
        return model

