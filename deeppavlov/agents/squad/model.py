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
from keras.optimizers import Adam
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

# Default parameters
default_opt = {}
default_opt['max_context_length'] = 300
default_opt['max_question_length'] = 30
default_opt['embedding_dim'] = 363
default_opt['learning_rate'] = 0.001
default_opt['batch_size'] = None
default_opt['epoch_num'] = 20
default_opt['seed'] = 42
default_opt['hidden_dim'] = 128
default_opt['dropout_val'] = 0.2
default_opt['recdrop_val'] = 0.0
default_opt['inpdrop_val'] = 0.2
default_opt['model_name'] = 'FastQA'

'''
--------------- Model ----------------------------
'''

class SquadModel(object):
    def __init__(self, opt = None, word_dict = None, feature_dict = None, weights_path = None ):
        if opt == None:
            opt = default_opt

        self.opt = copy.deepcopy(opt)
        self.word_dict = word_dict
        self.feature_dict = feature_dict

        self.max_context_length = None
        self.max_question_length = None
        self.embedding_dim = 300
        self.learning_rate = opt['learning_rate']
        self.epoch_num = opt['epoch_num']
        self.seed = opt['seed']
        self.hidden_dim = opt['hidden_dim']
        self.dropout_val = opt['dropout_val']
        self.recdrop_val = opt['recdrop_val']
        self.inpdrop_val = opt['inpdrop_val']
        self.model_name = opt['model_name']

        self.n_examples = 0
        self.updates = 0
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()

        self.model = self.fastqa_default()
        optimizer = Adam(lr=self.learning_rate)
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
        x, y = [batch[0], batch[3]], [cat(batch[5]), cat(batch[6])]

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

    def train_generator(self, train_data_gen, valid_data_gen, path):
        self.model.fit_generator(
            generator=train_data_gen,
            steps_per_epoch=train_data_gen.steps(),
            validation_data=valid_data_gen,
            validation_steps=valid_data_gen.steps(),
            epochs=self.epoch_num,
            callbacks=[
                ModelCheckpoint(path, verbose=1, save_best_only=True)
            ])

    def predict_generator(self, dev_data_gen):
        return self.model.predict_generator(generator=dev_data_gen,
                                            steps=dev_data_gen.steps(),
                                            verbose=1)

    def fastqa_default(self):
        '''Inputs'''
        print('Emb dim: ', self.embedding_dim)
        P = Input(shape=(self.max_context_length, self.embedding_dim), name='context_input')
        Q = Input(shape=(self.max_question_length, self.embedding_dim), name='question_input')

        passage_input = P
        question_input = Q

        '''Encoding'''
        encoder = Bidirectional(LSTM(units=self.hidden_dim,
                                     return_sequences=True,
                                     dropout=self.dropout_val,
                                     unroll=False))

        passage_encoding = passage_input
        passage_encoding = Masking()(passage_encoding)
        passage_encoding = encoder(passage_encoding)
        passage_encoding = projection(passage_encoding, self.embedding_dim)

        question_encoding = question_input
        question_encoding = Masking()(question_encoding)
        question_encoding = encoder(question_encoding)
        question_encoding = projection(question_encoding, self.embedding_dim)

        '''Attention over question'''
        question_attention_vector = question_attn_vector(question_encoding, passage_encoding)

        '''Answer span prediction'''
        # Answer start prediction
        answer_start = answer_start_pred(passage_encoding, question_attention_vector, self.embedding_dim)
        # Answer end prediction
        answer_end = answer_end_pred(passage_encoding, question_attention_vector, answer_start, self.embedding_dim)

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
