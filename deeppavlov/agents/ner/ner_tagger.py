import copy
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tensorflow.contrib.layers import xavier_initializer
import os
import pickle


class NERTagger:
    def __init__(self,
                 opt,
                 word_dict,
                 state_dict=None,
                 token_emb_dim=100,
                 char_emb_dim=25,
                 n_char_cnn_filters=25,
                 n_layers_per_block=4,
                 dilated_filter_width=3,
                 n_blocks=1,
                 learning_rate=1e-3):
        tf.reset_default_graph()
        self.token_emb_dim = token_emb_dim
        self.char_emb_dim = char_emb_dim
        self.n_char_cnn_filters = n_char_cnn_filters
        self.opt = copy.deepcopy(opt)
        vocab_size = len(word_dict)
        char_vocab_size = len(word_dict.char_dict)
        tag_vocab_size = len(word_dict.labels_dict)
        x_w = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
        x_c = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
        y_t = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')

        # Load embeddings
        w_embeddings = np.random.randn(vocab_size, token_emb_dim).astype(np.float32) / np.sqrt(token_emb_dim)
        c_embeddings = np.random.randn(char_vocab_size, char_emb_dim).astype(np.float32) / np.sqrt(char_emb_dim)
        w_embeddings = tf.Variable(w_embeddings, name='word_emb_var', trainable=True)
        c_embeddings = tf.Variable(c_embeddings, name='char_emb_var', trainable=True)

        # Word embedding layer
        w_emb = tf.nn.embedding_lookup(w_embeddings, x_w, name='word_emb')
        c_emb = tf.nn.embedding_lookup(c_embeddings, x_c, name='char_emb')

        # Character embedding network
        with tf.variable_scope('Char_Emb_Network'):
            char_filter_width = 3
            char_conv = tf.layers.conv2d(c_emb,
                                         n_char_cnn_filters,
                                         (1, char_filter_width),
                                         padding='same',
                                         name='char_conv')
            char_emb = tf.reduce_max(char_conv, axis=2)

        wc_features = tf.concat([w_emb, char_emb], axis=-1)

        # Cutdown dimensionality of the network via projection
        units = tf.layers.dense(wc_features, 50, kernel_initializer=xavier_initializer())

        units, auxilary_outputs = self.dense_network(units, n_layers_per_block, dilated_filter_width)

        logits = tf.layers.dense(units, tag_vocab_size, name='Dense')
        ground_truth_labels = tf.one_hot(y_t, tag_vocab_size, name='one_hot_tag_indxs')
        loss_tensor = tf.losses.softmax_cross_entropy(ground_truth_labels, logits)
        padding_mask = tf.cast(tf.not_equal(x_w, word_dict[word_dict.null_token]), tf.float32)
        loss_tensor = loss_tensor * padding_mask
        loss = tf.reduce_mean(loss_tensor)

        self.loss = loss
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        self.sess = tf.Session()
        self.word_dict = word_dict
        self.x = x_w
        self.xc = x_c
        self.y_ground_truth = y_t
        self.y_predicted = tf.argmax(logits, axis=2)
        if self.opt.get('pretrained_model'):
            self.load(self.opt.get('pretrained_model'))
        else:
            self.sess.run(tf.global_variables_initializer())

    def dense_network(self, units, n_layers, filter_width):
        n_filters = units.get_shape().as_list()[-1]
        outputs = [units]
        auxilary_outputs = []
        for n_layer in range(n_layers):
            if len(outputs) > 1:
                units = tf.concat(outputs, axis=-1)
            else:
                units = outputs[0]
            units = tf.layers.conv1d(units,
                                     n_filters,
                                     filter_width,
                                     padding='same',
                                     name='Layer_' + str(n_layer),
                                     activation=None,
                                     kernel_initializer=xavier_initializer())
            auxilary_outputs.append(units)
            units = tf.nn.relu(units)
            outputs.append(units)
        units = tf.concat(outputs, axis=-1)
        return units, auxilary_outputs

    def character_embedding_network(self, x_char, n_filters, filter_width):
        pass

    def train_on_batch(self, x, xc, y):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.x: x, self.xc: xc, self.y_ground_truth: y})
        return loss

    def eval(self, x, y):
        loss = self.sess.run(self.loss, feed_dict={self.x: x, self.y_ground_truth: y})
        return loss

    def predict(self, x, xc):
        y = self.sess.run(self.y_predicted, feed_dict={self.x: x, self.xc: xc})
        return y

    def save(self, file_path):
        saver = tf.train.Saver()
        print('saving path ' + os.path.join(file_path, 'model.ckpt'))
        saver.save(self.sess, os.path.join(file_path, 'model.ckpt'))

    def load(self, file_path):
        saver = tf.train.Saver()
        print('loading path ' + os.path.join(file_path, 'model.ckpt'))
        saver.restore(self.sess, os.path.join(file_path, 'model.ckpt'))

    def shutdown(self):
        tf.reset_default_graph()
