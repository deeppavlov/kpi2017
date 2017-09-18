import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

from .metrics import roc_auc_score
import os
import numpy as np
import copy
from keras.layers import Dense, Activation, Input, concatenate
from keras.models import Model

from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout, Reshape
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn import linear_model, svm
from keras import backend as K
from keras.metrics import binary_accuracy
import json
import pickle
from .utils import vectorize_select_from_data

from .embeddings_dict import EmbeddingsDict

class InsultsModel(object):

    def __init__(self, model_name, word_index, embedding_dict, opt):
        self.model_name = model_name
        self.word_index = word_index
        self.embedding_dict = None
        self.opt = copy.deepcopy(opt)
        self.kernel_sizes = [int(x) for x in opt['kernel_sizes_cnn'].split(' ')]
        self.pool_sizes = [int(x) for x in opt['pool_sizes_cnn'].split(' ')]
        self.model_type = None
        self.from_saved = False

        if self.model_name == 'cnn_word' or self.model_name == 'lstm_word':
            self.model_type = 'nn'
            self.embedding_dict = embedding_dict if embedding_dict is not None else EmbeddingsDict(opt, self.opt['embedding_dim'])

        if self.model_name == 'log_reg' or self.model_name == 'svc':
            self.model_type = 'ngrams'
            self.num_ngrams = None
            self.vectorizers = None
            self.selectors = None

        if self.opt.get('model_file') and \
                ( (os.path.isfile(opt['model_file'] + '.h5') and self.model_type == 'nn')
                 or (os.path.isfile(opt['model_file'] + '_opt.json') and
                     os.path.isfile(opt['model_file'] + '_cls.pkl') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_special.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_0.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_1.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_2.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_3.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_4.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_5.bin') and
                             self.model_type == 'ngrams') ):
            print('[Initializing model from saved]')
            self.from_saved = True
            self._init_from_saved(opt['model_file'])
        else:
            if self.opt.get('pretrained_model'):
                print('[Initializing model from pretrained]')
                self.from_saved = True
                self._init_from_saved(opt['pretrained_model'])
            else:
                print('[ Initializing model from scratch ]')
                self._init_from_scratch()

        self.opt['cuda'] = not self.opt['no_cuda']

        self.n_examples = 0
        self.updates = 0
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.train_auc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.val_auc = 0.0


    def _init_from_scratch(self):
        if self.model_name == 'log_reg':
            self.model = self.log_reg_model()
        if self.model_name == 'svc':
            self.model = self.svc_model()
        if self.model_name == 'cnn_word':
            self.model = self.cnn_word_model()
        if self.model_name == 'lstm_word':
            self.model = self.lstm_word_model()

        if self.model_type == 'nn':
            optimizer = Adam(lr=self.opt['learning_rate'], decay=self.opt['learning_decay'])
            self.model.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['binary_accuracy'])

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname

        if fname:
            if self.model_type == 'nn':
                print("[ saving model: " + fname + " ]")
                self.model.save_weights(fname + '.h5')
                self.embedding_dict.save_items(fname)

            if self.model_type == 'ngrams':
                print("[ saving model: " + fname + " ]")
                with open(fname + '_cls.pkl', 'wb') as model_file:
                    pickle.dump(self.model, model_file)

            with open(fname + '_opt.json', 'w') as opt_file:
                json.dump(self.opt, opt_file)

    def _init_from_saved(self, fname):

        with open(fname + '_opt.json', 'r') as opt_file:
            self.opt = json.load(opt_file)

        if self.model_type == 'nn':
            if self.model_name == 'cnn_word':
                self.model = self.cnn_word_model()
            if self.model_name == 'lstm_word':
                self.model = self.lstm_word_model()

            optimizer = Adam(lr=self.opt['learning_rate'], decay=self.opt['learning_decay'])
            self.model.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['binary_accuracy'])
            print('[ Loading model weights %s ]' % fname)
            self.model.load_weights(fname + '.h5')

        if self.model_type == 'ngrams':
            with open(fname + '_cls.pkl', 'rb') as model_file:
                self.model = pickle.load(model_file)
            print('CLS:', self.model)

    def _build_ex(self, ex):
        if 'text' not in ex:
            return
        """Find the token span of the answer in the context for this example.
        """
        inputs = dict()
        inputs['question'] = ex['text']
        if 'labels' in ex:
            inputs['labels'] = ex['labels']

        return inputs

    def _predictions2text(self, predictions):
        y = ['Insult' if ex > 0.5 else 'Non-insult' for ex in predictions]
        return y

    def _text2predictions(self, predictions):
        y = [1 if ex == 'Insult' else 0 for ex in predictions]
        return y

    def _batchify(self, batch, word_dict=None):
        if self.model_type == 'nn':
            question = []
            for ex in batch:
                question.append(ex['question'])
            self.embedding_dict.add_items(question)
            embedding_batch = self.create_batch(question)

            if len(batch[0]) == 2:
                y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
                return embedding_batch, y
            else:
                return embedding_batch

        if self.model_type == 'ngrams':
            question = []
            for ex in batch:
                question.append(ex['question'])

            if len(batch[0]) == 2:
                y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
                return question, y
            else:
                return question

    def create_batch(self, sentence_li):
        embeddings_batch = []
        for sen in sentence_li:
            embeddings = []
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            if len(tokens) > self.opt['max_sequence_length']:
                tokens = tokens[:self.opt['max_sequence_length']]
            for tok in tokens:
                embeddings.append(self.embedding_dict.tok2emb.get(tok))
            if len(tokens) < self.opt['max_sequence_length']:
                pads = [np.zeros(self.opt['embedding_dim'])
                        for _ in range(self.opt['max_sequence_length'] - len(tokens))]
                embeddings = pads + embeddings
            embeddings = np.asarray(embeddings)
            embeddings_batch.append(embeddings)
        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def update(self, batch):
        x, y = batch
        y = np.array(y)
        y_pred = None

        if self.model_type == 'nn':
            self.train_loss, self.train_acc = self.model.train_on_batch(x, y)
            y_pred = self.model.predict_on_batch(x).reshape(-1)
            self.train_auc = roc_auc_score(y, y_pred)

        if self.model_type == 'ngrams':
            x = vectorize_select_from_data(x, self.vectorizers, self.selectors)
            self.model.fit(x, y.reshape(-1))
            y_pred = np.array(self.model.predict_proba(x)[:,1]).reshape(-1)
            y_pred_tensor = K.constant(y_pred, dtype='float64')
            self.train_loss = K.eval(binary_crossentropy(y.astype('float'), y_pred_tensor))
            self.train_acc = K.eval(binary_accuracy(y.astype('float'), y_pred_tensor))
            self.train_auc = roc_auc_score(y, y_pred)
        self.updates += 1
        return y_pred

    def predict(self, batch):
        if self.model_type == 'nn':
            y_pred = np.array(self.model.predict_on_batch(batch)).reshape(-1)
            return y_pred
        if self.model_type == 'ngrams':
            x = vectorize_select_from_data(batch, self.vectorizers, self.selectors)
            predictions = np.array(self.model.predict_proba(x)[:,1]).reshape(-1)
            return predictions

    def shutdown(self):
        self.embedding_dict = None

    def log_reg_model(self):
        model = linear_model.LogisticRegression(C=10.)
        return model

    def svc_model(self):
        model = svm.SVC(probability=True, C=0.3, kernel='linear')
        return model

    def cnn_word_model(self):
        embed_input = Input(shape=(self.opt['max_sequence_length'], self.opt['embedding_dim'],))

        outputs = []
        for i in range(len(self.kernel_sizes)):
            output_i = Conv1D(self.opt['filters_cnn'], kernel_size=self.kernel_sizes[i], activation='relu',
                              kernel_regularizer=l2(self.opt['regul_coef_conv']), padding='same')(embed_input)
            output_i = MaxPooling1D(pool_size=self.pool_sizes[i], strides=1, padding='same')(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)
        output = Reshape(((self.opt['max_sequence_length']
                           * len(self.kernel_sizes))
                          * self.opt['filters_cnn'],))(output)
        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.opt['dense_dim'], activation='relu',
                       kernel_regularizer=l2(self.opt['regul_coef_dense']))(output)
        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(1, activation=None, kernel_regularizer=l2(self.opt['regul_coef_dense']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=embed_input, outputs=act_output)
        return model

    def lstm_word_model(self):
        embed_input = Input(shape=(self.opt['max_sequence_length'], self.opt['embedding_dim'],))

        output_0 = Bidirectional(LSTM(self.opt['units_lstm'],
                                      kernel_regularizer=l2(self.opt['regul_coef_lstm']),
                                      dropout=self.opt['dropout_rate']))(embed_input)

        output_1 = Bidirectional(LSTM(self.opt['units_lstm'],
                                      kernel_regularizer=l2(self.opt['regul_coef_lstm']),
                                      dropout=self.opt['dropout_rate']))(embed_input)

        output_2 = Bidirectional(LSTM(self.opt['units_lstm'],
                                      kernel_regularizer=l2(self.opt['regul_coef_lstm']),
                                      dropout=self.opt['dropout_rate']))(embed_input)

        output = concatenate([output_0, output_1, output_2], axis=1)

        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.opt['dense_dim'], activation='relu',
                       kernel_regularizer=l2(self.opt['regul_coef_dense']))(output)
        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(1, activation=None,
                       kernel_regularizer=l2(self.opt['regul_coef_dense']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=embed_input, outputs=act_output)
        return model


