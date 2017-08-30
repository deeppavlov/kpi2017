# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import json, os, copy
import pickle
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda,  multiply, Masking
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.activations import softmax as Softmax
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint
from .utils import AverageMeter

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

        for k, v in opt.items():
            setattr(self, k, v)

        self.word_dict = word_dict
        self.feature_dict = feature_dict

        self.n_examples = 0
        self.updates = 0
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()

        self.model = self.fastqa_default()
        optimizer = Adamax(decay=0.0)

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

        x, y = [np.concatenate((batch[0], batch[1]), axis=2), batch[3]], [cat(batch[5]), cat(batch[6])]

        output = self.model.train_on_batch(x, y)
        self.train_loss.update(output[0])
        self.train_acc.update((output[3] + output[4])/2)
        self.updates += 1

    def predict(self, batch):

        score_s, score_e = self.model.predict_on_batch([batch[0], batch[3]])
        s_ind, e_ind = np.argmax(score_s, axis=1), np.argmax(score_e, axis=1)
        # Get argmax text spans
        text = batch[-2]
        spans = batch[-1]
        predictions = []

        # Return predictions in the form of text
        max_len = score_s.shape[1]

        for i in range(score_s.shape[0]):
            if s_ind[i] < e_ind[i]:
                s_offset, e_offset = spans[i][s_ind[i]][0], spans[i][e_ind[i]][1]
                predictions.append(text[i][s_offset:e_offset])
            else:
                predictions.append(' ')

        return predictions


    def build_ex(self, ex):
        if 'text' not in ex:
            return

        """Find the token span of the answer in the context for this example.
        """
        inputs = dict()
        texts = ex['text'].split('\n')
        inputs['context'] = texts[1]
        inputs['question'] = texts[2]
        if 'labels' in ex:
            inputs['answer'] = ex['labels']

        return inputs


    def fastqa_default(self):

        '''Inputs'''
        P = Input(shape=(None, self.context_embedding_dim), name='context_input')
        Q = Input(shape=(None, self.question_embedding_dim), name='question_input')

        passage_input = P
        question_input = Q

        '''Encoding'''
        passage_encoding = passage_input
        passage_encoding = Masking()(passage_encoding)
        passage_encoding = biLSTM_encoder(
            passage_encoding,
            self.encoder_hidden_dim,
            self.rnn_dropout,
            self.recurrent_dropout,
            self.question_enc_layers)
        passage_encoding = projection(passage_encoding, self.projection_dim)

        question_encoding = question_input
        question_encoding = Masking()(question_encoding)
        question_encoding = biLSTM_encoder(
            question_encoding,
            self.encoder_hidden_dim,
            self.rnn_dropout,
            self.recurrent_dropout,
            self.context_enc_layers)
        question_encoding = projection(question_encoding, self.projection_dim)

        '''Attention over question'''
        question_attention_vector = question_attn_vector(question_encoding, passage_encoding)

        '''Answer span prediction'''
        # Answer start prediction
        answer_start = answer_start_pred(passage_encoding, question_attention_vector, self.pointer_dim)
        # Answer end prediction
        answer_end = answer_end_pred(passage_encoding, question_attention_vector, answer_start, self.pointer_dim)

        input_placeholders = [P, Q]
        inputs = input_placeholders
        outputs = [answer_start, answer_end]

        model = Model(inputs=inputs, outputs=outputs)
        return model

if __name__ == "__main__":

    # Default parameters
    opt = {}
    opt['max_context_length'] = 768
    opt['max_question_length'] = 100
    opt['embedding_dim'] = 300
    opt['learning_rate'] = 0.001
    opt['batch_size'] = None
    opt['epoch_num'] = 20
    opt['seed'] = 42
    opt['hidden_dim'] = 128
    opt['dropout_val'] = 0.2
    opt['recdrop_val'] = 0.0
    opt['inpdrop_val'] = 0.2
    opt['model_name'] = 'FastQA'

    # Initializing model
    fasqa = SquadModel(opt)
