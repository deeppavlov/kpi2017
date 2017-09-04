import copy
import tensorflow as tf
import numpy as np
from collections import namedtuple


# from torch.multiprocessing.pool import Pool
from .dictionary import NERDictionaryAgent

POS_Tagger_State = namedtuple('POS_Tagger_State', 'words char_embs input_index words_count prev_pos output terminated')


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

        self.token_emb_dim = token_emb_dim
        self.char_emb_dim = char_emb_dim
        self.n_char_cnn_filters = n_char_cnn_filters
        self.opt = copy.deepcopy(opt)
        vocab_size = len(word_dict)
        char_vocab_size = len(word_dict.char_dict)
        tag_vocab_size = len(word_dict.labels_dict)

        if state_dict:
            self.load_state_dict(state_dict)

        x_w = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
        x_c = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
        y_t = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')

        # Load embeddings
        w_embeddings = np.random.randn(vocab_size, token_emb_dim).astype(np.float32) / np.sqrt(token_emb_dim)
        c_embeddings = np.random.randn(char_vocab_size, char_emb_dim).astype(np.float32) / np.sqrt(char_emb_dim)

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
        n_filters_dilated = token_emb_dim

        units = wc_features
        # Stack blocks of dilated layers with shared weights (all blocks share weights)
        for n_block in range(n_blocks):

            reuse_layer = n_block > 0
            for n_layer in range(n_layers_per_block):
                units = tf.layers.conv1d(units,
                                         n_filters_dilated,
                                         dilated_filter_width,
                                         padding='same',
                                         name='Layer_' + str(n_layer),
                                         reuse=reuse_layer,
                                         activation=None)
                units = tf.nn.relu(units)

        logits = tf.layers.dense(units, tag_vocab_size, name='Dense')
        ground_truth_labels = tf.one_hot(y_t, tag_vocab_size, name='one_hot_tag_indxs')
        loss_tensor = tf.losses.softmax_cross_entropy(ground_truth_labels, logits)
        padding_mask = tf.cast(tf.not_equal(x_w, word_dict[word_dict.null_token]), tf.float32)
        loss_tensor = loss_tensor * padding_mask
        loss = tf.reduce_mean(loss_tensor)

        self.loss = loss
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.x = x_w
        self.xc = x_c
        self.y_ground_truth = y_t
        self.y_predicted = tf.argmax(logits)

    def character_embedding_network(self, x_char, n_filters, filter_width):
        pass

    def train_on_batch(self, x, xc, y):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.x: x, self.xc: xc, self.y_ground_truth: y})
        return loss

    def eval(self, x, y):
        loss = self.sess.run(self.loss, feed_dict={self.x: x, self.y_ground_truth: y})
        return loss

    def predict(self, x):
        y = self.sess.run(self.y_predicted, feed_dict={self.x: x})
        return y

    def save(self, file_path):
        saver = tf.train.Saver()
        saver.save(self.sess, file_path)

    def load(self, file_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, file_path)

    def shutdown(self):
        tf.reset_default_graph()
