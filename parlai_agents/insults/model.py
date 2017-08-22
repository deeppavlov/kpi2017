from .metrics import roc_auc_score

import os
import numpy as np
import copy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from keras.layers import Dense, Activation, Input, Embedding, concatenate
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Reshape
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn import linear_model, svm
from keras import backend as K

class InsultsModel(object):

    def __init__(self, model_name, word_index, embedding_matrix, opt):
        self.model_name = model_name
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
        self.opt = copy.deepcopy(opt)
        self.max_sequence_length = opt['max_sequence_length']
        self.embedding_dim = opt['embedding_dim']
        self.learning_rate = opt['learning_rate']
        self.learning_decay = opt['learning_decay']
        self.seed = opt['seed']
        self.num_filters = opt['num_filters']
        self.kernel_sizes = [int(x) for x in opt['kernel_sizes'].split(' ')]
        self.regul_coef_conv = opt['regul_coef_conv']
        self.regul_coef_dense = opt['regul_coef_dense']
        self.pool_sizes = [int(x) for x in opt['pool_sizes'].split(' ')]
        self.dropout_rate = opt['dropout_rate']
        self.dense_dim = opt['dense_dim']
        self.model_type = None
        if self.model_name == 'cnn_word':
            self.model_type = 'nn'
        if self.model_name == 'log_reg' or self.model_name == 'svc':
            self.model_type = 'ngrams'

        if self.opt.get('model_file') and os.path.isfile(opt['model_file']):
            self._init_from_saved(opt['model_file'])
        else:
            if self.opt.get('pretrained_model'):
                self._init_from_saved(opt['pretrained_model'])
            else:
                self._init_from_scratch()
        self.opt['cuda'] = not self.opt['no_cuda']
        # if self.opt['cuda']:
        #     print('[ Using CUDA (GPU %d) ]' % opt['gpu'])
        #     config = tf.ConfigProto()
        #     config.gpu_options.per_process_gpu_memory_fraction = 0.45
        #     config.gpu_options.visible_device_list = str(opt['gpu'])
        #     set_session(tf.Session(config=config))

        self.n_examples = 0
        self.updates = 0
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.train_auc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.val_auc = 0.0


    def _init_from_scratch(self):
        print('[ Initializing model from scratch ]')
        if self.model_name == 'log_reg':
            self.model = self.log_reg_model()
        if self.model_name == 'svc':
            self.model = self.svc_model()
        if self.model_name == 'cnn_word':
            self.model = self.cnn_word_model()

        if self.model_type == 'nn':
            optimizer = Adam(lr=self.learning_rate, decay=self.learning_decay)
            self.model.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname

        if fname:
            if self.model_type == 'nn':
                print("[ saving model: " + fname + " ]")
                self.model.save(fname + '.h5')

            if self.model_type == 'ngrams':
                print("[ saving model: " + fname + " ]")
                best_params = list(self.model.best_params_.items())
                f = open(fname + '.txt', 'w')
                for i in range(len(best_params)):
                    f.write(best_params[i][0] + ' ' + str(best_params[i][1]) + '\n')
                f.close()

    def _init_from_saved(self, fname):
        print('[ Loading model %s ]' % fname)
        if self.model_type == 'nn':
            self.model = load_model(fname + '.h5')
        if self.model_type == 'ngrams':
            f = open(fname + '.txt', 'r')
            data = f.readlines()
            best_params = dict()
            for i in range(len(data)):
                line_split =data[i][:-1].split(' ')
                try:
                    best_params[line_split[0]] = float(line_split[1])
                except ValueError:
                    best_params[line_split[0]] = line_split[1]
            self.model_best_params_ = best_params
            f.close()

    def update(self, batch):
        x, y = batch
        #x = np.array(x)
        y = np.array(y)
        if self.model_type == 'nn':
            self.train_loss, self.train_acc = self.model.train_on_batch(x, y)
            y_pred = self.model.predict_on_batch(x).reshape(-1)
            self.train_auc = roc_auc_score(y, y_pred)
        if self.model_type == 'ngrams':
            self.model.fit(x, y)
            y_pred = self.model.predict_proba(x).reshape(-1)

            self.train_loss = binary_crossentropy(y, y_pred),
            self.train_acc = accuracy(y, y_pred)
            self.train_auc = roc_auc_score(y, y_pred)
        self.updates += 1

    def predict(self, batch):
        if self.model_type == 'nn':
            return self.model.predict_on_batch(batch)
        if self.model_type == 'ngrams':
            return self.model.predict_proba(batch)

    def log_reg_model(self):
        model = linear_model.LogisticRegression()
        params = {'C': [10]}
        best_model = GridSearchCV(model, params, cv=self.kfold)
        return best_model

    def svc_model(self):
        model = svm.SVC(probability=True)
        params = {'C': [0.5], 'kernel': 'linear'}
        best_model = GridSearchCV(model, params, cv=self.kfold)
        return best_model

    def cnn_word_model(self):

        input = Input(shape=(self.max_sequence_length,))
        embed_input = Embedding(len(self.word_index) + 1, self.embedding_dim,
                                weights=[self.embedding_matrix] if self.embedding_matrix is not None else None,
                                input_length=self.max_sequence_length,
                                trainable=self.embedding_matrix is None)(input)

        output_0 = Conv1D(self.num_filters, kernel_size=self.kernel_sizes[0], activation='relu',
                          kernel_regularizer=l2(self.regul_coef_conv), padding='same')(embed_input)
        output_0 = MaxPooling1D(pool_size=self.pool_sizes[0], strides=1, padding='same')(output_0)

        output_1 = Conv1D(self.num_filters, kernel_size=self.kernel_sizes[1], activation='relu',
                          kernel_regularizer=l2(self.regul_coef_conv), padding='same')(embed_input)
        output_1 = MaxPooling1D(pool_size=self.pool_sizes[1], strides=1, padding='same')(output_1)

        output_2 = Conv1D(self.num_filters, kernel_size=self.kernel_sizes[2], activation='relu',
                          kernel_regularizer=l2(self.regul_coef_conv), padding='same')(embed_input)
        output_2 = MaxPooling1D(pool_size=self.pool_sizes[2], strides=1, padding='same')(output_2)
        output = concatenate([output_0, output_1, output_2], axis=1)
        output = Reshape(((self.max_sequence_length * len(self.kernel_sizes)) * self.num_filters,))(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(self.dense_dim, activation='relu', kernel_regularizer=l2(self.regul_coef_dense))(output)
        output = Dropout(rate=self.dropout_rate)(output)
        output = Dense(1, activation=None, kernel_regularizer=l2(self.regul_coef_dense))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=input, outputs=act_output)
        return model
