"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random
import os
import threading
import copy
import numpy as np
import tensorflow as tf
from . import utils
from os.path import isdir

# coref_op_library = tf.load_op_library("./coref_kernels.so")
coref_op_library = tf.load_op_library('/home/petrov/coreference_kpi/coreference/coref_kernels.so') # Need path as variable in opt 
spans = coref_op_library.spans
tf.NotDifferentiable("Spans")
get_antecedents = coref_op_library.antecedents
tf.NotDifferentiable("Antecedents")
extract_mentions = coref_op_library.extract_mentions
tf.NotDifferentiable("ExtractMentions")
distance_bins = coref_op_library.distance_bins
tf.NotDifferentiable("DistanceBins")

class CorefModel(object):
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)

        self.embedding_info = (self.opt["embedding_size"], self.opt["emb_lowercase"])
        self.embedding_size = self.opt['embedding_size']
        self.char_embedding_size = self.opt["char_embedding_size"]
        self.char_dict = utils.load_char_dict(self.opt["char_vocab_path"])
        self.embedding_dicts = utils.load_embedding_dict(self.opt["embedding_path"], self.embedding_size, self.opt["emb_format"])
        self.max_mention_width = self.opt["max_mention_width"]
        self.genres = {g: i for i, g in enumerate(self.opt["genres"])}

        input_props = []
        input_props.append((tf.float32, [None, None, self.embedding_size]))  # Text embeddings.
        input_props.append((tf.int32, [None, None, None]))  # Character indices.
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None]))  # Speaker IDs.
        input_props.append((tf.int32, []))  # Genre.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # Gold starts.
        input_props.append((tf.int32, [None]))  # Gold ends.
        input_props.append((tf.int32, [None]))  # Cluster ids.

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=1, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()
        
        
        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.opt["learning_rate"], self.global_step,
                                                   self.opt["decay_frequency"], self.opt["decay_rate"],
                                                   staircase=True)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, self.opt["max_gradient_norm"])
        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.opt["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def start_enqueue_thread(self, train_example, is_training):
        def _enqueue_loop():
            while True:
                tensorized_example = self.tensorize_example(train_example, is_training=is_training)
                feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                self.sess.run(self.enqueue_op, feed_dict=feed_dict)
        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_example(self, example, is_training, oov_counts=None):
        clusters = example["clusters"]
        gold_mentions = sorted(tuple(m) for m in utils.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = utils.flatten(example["speakers"])

        assert num_words == len(speakers)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.opt["filter_widths"]))
        word_emb = np.zeros([len(sentences), max_sentence_length, self.embedding_size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        text_len = np.array([len(s) for s in sentences])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                current_dim = 0
                d = self.embedding_dicts
                (s, l) = self.embedding_info
                if l:
                    cerrent_word = word.lower()
                else:
                    current_word = word
#                for (d, (s, l)) in zip(self.embedding_dicts, self.embedding_info):
#                    if l:
#                        current_word = word.lower()
#                    else:
#                        current_word = word
#                    if oov_counts is not None and current_word not in d:
#                        oov_counts[0] += 1
#                    print(type(d), current_word, type(current_word))
                word_emb[i, j, current_dim:current_dim + s] = utils.normalize(d[current_word])
                current_dim += s
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])  # numpy

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]  # int 1

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)  # numpy of unicode str

        # print 'gold_ends_len', len(gold_ends), type(gold_ends[0])
        # print 'gold_starts_len', len(gold_starts), type(gold_starts[0])

        if is_training and len(sentences) > self.opt["max_training_sentences"]:
            return self.truncate_example(word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts,
                                         gold_ends, cluster_ids)
        else:
            return word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids

    def truncate_example(self, word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                         cluster_ids):
        max_training_sentences = self.opt["max_training_sentences"]
        num_sentences = word_emb.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences)

        # print 'sentence_offset', type(sentence_offset), sentence_offset

        word_offset = text_len[:sentence_offset].sum()

        # print 'word_offset', type(word_offset), word_offset

        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        word_emb = word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        speaker_ids = speaker_ids[word_offset: word_offset + num_words]

        assert len(gold_ends) == len(gold_starts)
        Gold_starts = np.zeros((len(gold_starts)))
        Gold_ends = np.zeros((len(gold_ends)))
        for i in range(len(gold_ends)):
            Gold_ends[i] = int(gold_ends[i])
            Gold_starts[i] = int(gold_starts[i])
        gold_starts = Gold_starts
        gold_ends = Gold_ends
        # here hernya
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)

        # print 'gold_spans', type(gold_spans), gold_spans
        # print 'gold_starts[gold_spans]', type(gold_starts[False]), gold_starts[False]

        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset

        cluster_ids = cluster_ids[gold_spans]

        return word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids

    def get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
        mention_emb_list = []

        mention_start_emb = tf.gather(text_outputs, mention_starts)  # [num_mentions, emb]
        mention_emb_list.append(mention_start_emb)

        mention_end_emb = tf.gather(text_outputs, mention_ends)  # [num_mentions, emb]
        mention_emb_list.append(mention_end_emb)

        mention_width = 1 + mention_ends - mention_starts  # [num_mentions]
        if self.opt["use_features"]:
            mention_width_index = mention_width - 1  # [num_mentions]
            mention_width_emb = tf.gather(tf.get_variable("mention_width_embeddings", [self.opt["max_mention_width"],
                                                                                       self.opt["feature_size"]]),
                                          mention_width_index)  # [num_mentions, emb]
            mention_width_emb = tf.nn.dropout(mention_width_emb, self.dropout)
            mention_emb_list.append(mention_width_emb)

        if self.opt["model_heads"]:
            mention_indices = tf.expand_dims(tf.range(self.opt["max_mention_width"]), 0) + tf.expand_dims(
                mention_starts, 1)  # [num_mentions, max_mention_width]
            mention_indices = tf.minimum(utils.shape(text_outputs, 0) - 1,
                                         mention_indices)  # [num_mentions, max_mention_width]
            mention_text_emb = tf.gather(text_emb, mention_indices)  # [num_mentions, max_mention_width, emb]
            self.head_scores = utils.projection(text_outputs, 1)  # [num_words, 1]
            mention_head_scores = tf.gather(self.head_scores, mention_indices)  # [num_mentions, max_mention_width, 1]
            mention_mask = tf.expand_dims(
                tf.sequence_mask(mention_width, self.opt["max_mention_width"], dtype=tf.float32),
                2)  # [num_mentions, max_mention_width, 1]
            mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask),
                                              dim=1)  # [num_mentions, max_mention_width, 1]
            mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1)  # [num_mentions, emb]
            mention_emb_list.append(mention_head_emb)

        mention_emb = tf.concat(mention_emb_list, 1)  # [num_mentions, emb]
        return mention_emb

    def get_mention_scores(self, mention_emb):
        with tf.variable_scope("mention_scores"):
            return utils.ffnn(mention_emb, self.opt["ffnn_depth"], self.opt["ffnn_size"], 1,
                             self.dropout)  # [num_mentions, 1]

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [num_mentions, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [num_mentions]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [num_mentions]
        return log_norm - marginalized_gold_scores  # [num_mentions]

    def get_antecedent_scores(self, mention_emb, mention_scores, antecedents, antecedents_len, mention_starts,
                              mention_ends, mention_speaker_ids, genre_emb):
        num_mentions = utils.shape(mention_emb, 0)
        max_antecedents = utils.shape(antecedents, 1)

        feature_emb_list = []

        if self.opt["use_metadata"]:
            antecedent_speaker_ids = tf.gather(mention_speaker_ids, antecedents)  # [num_mentions, max_ant]
            same_speaker = tf.equal(tf.expand_dims(mention_speaker_ids, 1),
                                    antecedent_speaker_ids)  # [num_mentions, max_ant]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.opt["feature_size"]]),
                                         tf.to_int32(same_speaker))  # [num_mentions, max_ant, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0),
                                      [num_mentions, max_antecedents, 1])  # [num_mentions, max_ant, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.opt["use_features"]:
            target_indices = tf.range(num_mentions)  # [num_mentions]
            mention_distance = tf.expand_dims(target_indices, 1) - antecedents  # [num_mentions, max_ant]
            mention_distance_bins = distance_bins(mention_distance)  # [num_mentions, max_ant]
            mention_distance_bins.set_shape([None, None])
            mention_distance_emb = tf.gather(tf.get_variable("mention_distance_emb", [10, self.opt["feature_size"]]),
                                             mention_distance_bins)  # [num_mentions, max_ant]
            feature_emb_list.append(mention_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [num_mentions, max_ant, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [num_mentions, max_ant, emb]

        antecedent_emb = tf.gather(mention_emb, antecedents)  # [num_mentions, max_ant, emb]
        target_emb_tiled = tf.tile(tf.expand_dims(mention_emb, 1),
                                   [1, max_antecedents, 1])  # [num_mentions, max_ant, emb]
        similarity_emb = antecedent_emb * target_emb_tiled  # [num_mentions, max_ant, emb]

        pair_emb = tf.concat([target_emb_tiled, antecedent_emb, similarity_emb, feature_emb],
                             2)  # [num_mentions, max_ant, emb]

        with tf.variable_scope("iteration"):
            with tf.variable_scope("antecedent_scoring"):
                antecedent_scores = utils.ffnn(pair_emb, self.opt["ffnn_depth"], self.opt["ffnn_size"], 1,
                                              self.dropout)  # [num_mentions, max_ant, 1]
        antecedent_scores = tf.squeeze(antecedent_scores, 2)  # [num_mentions, max_ant]

        antecedent_mask = tf.log(
            tf.sequence_mask(antecedents_len, max_antecedents, dtype=tf.float32))  # [num_mentions, max_ant]
        antecedent_scores += antecedent_mask  # [num_mentions, max_ant]

        antecedent_scores += tf.expand_dims(mention_scores, 1) + tf.gather(mention_scores,
                                                                           antecedents)  # [num_mentions, max_ant]
        antecedent_scores = tf.concat([tf.zeros([utils.shape(mention_scores, 0), 1]), antecedent_scores],
                                      1)  # [num_mentions, max_ant + 1]
        return antecedent_scores  # [num_mentions, max_ant + 1]


    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, utils.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, text_len_mask)

    def encode_sentences(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(text_emb)[0]
        max_sentence_length = tf.shape(text_emb)[1]

        # Transpose before and after for efficiency.
        inputs = tf.transpose(text_emb, [1, 0, 2])  # [max_sentence_length, num_sentences, emb]

        with tf.variable_scope("fw_cell"):
            cell_fw = utils.CustomLSTMCell(self.opt["lstm_size"], num_sentences, self.dropout)
            preprocessed_inputs_fw = cell_fw.preprocess_input(inputs)
        with tf.variable_scope("bw_cell"):
            cell_bw = utils.CustomLSTMCell(self.opt["lstm_size"], num_sentences, self.dropout)
            preprocessed_inputs_bw = cell_bw.preprocess_input(inputs)
            preprocessed_inputs_bw = tf.reverse_sequence(preprocessed_inputs_bw,
                                                         seq_lengths=text_len,
                                                         seq_dim=0,
                                                         batch_dim=1)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                 tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                 tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))
        with tf.variable_scope("lstm"):
            with tf.variable_scope("fw_lstm"):
                fw_outputs, fw_states = tf.nn.dynamic_rnn(cell=cell_fw,
                                                          inputs=preprocessed_inputs_fw,
                                                          sequence_length=text_len,
                                                          initial_state=state_fw,
                                                          time_major=True)
            with tf.variable_scope("bw_lstm"):
                bw_outputs, bw_states = tf.nn.dynamic_rnn(cell=cell_bw,
                                                          inputs=preprocessed_inputs_bw,
                                                          sequence_length=text_len,
                                                          initial_state=state_bw,
                                                          time_major=True)

        bw_outputs = tf.reverse_sequence(bw_outputs,
                                         seq_lengths=text_len,
                                         seq_dim=0,
                                         batch_dim=1)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2)
        text_outputs = tf.transpose(text_outputs, [1, 0, 2])  # [num_sentences, max_sentence_length, emb]
        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents


    def get_predictions_and_loss(self, word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts,
                                 gold_ends, cluster_ids):
        self.dropout = 1 - (tf.to_float(is_training) * self.opt["dropout_rate"])
        self.lexical_dropout = 1 - (tf.to_float(is_training) * self.opt["lexical_dropout_rate"])

        num_sentences = tf.shape(word_emb)[0]
        max_sentence_length = tf.shape(word_emb)[1]

        text_emb_list = [word_emb]

        if self.opt["char_embedding_size"] > 0:
            char_emb = tf.gather(
                tf.get_variable("char_embeddings", [len(self.char_dict), self.opt["char_embedding_size"]]),
                char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, utils.shape(char_emb, 2),
                                                       utils.shape(char_emb,
                                                                  3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
            flattened_aggregated_char_emb = utils.cnn(flattened_char_emb, self.opt["filter_widths"], self.opt[
                "filter_size"])  # [num_sentences * max_sentence_length, emb]
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                             utils.shape(flattened_aggregated_char_emb,
                                                                                        1)])  # [num_sentences, max_sentence_length, emb]
            text_emb_list.append(aggregated_char_emb)

        text_emb = tf.concat(text_emb_list, 2)
        text_emb = tf.nn.dropout(text_emb, self.lexical_dropout)

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)
        text_len_mask = tf.reshape(text_len_mask, [num_sentences * max_sentence_length])

        text_outputs = self.encode_sentences(text_emb, text_len, text_len_mask)
        text_outputs = tf.nn.dropout(text_outputs, self.dropout)

        genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.opt["feature_size"]]),
                              genre)  # [emb]

        sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1),
                                   [1, max_sentence_length])  # [num_sentences, max_sentence_length]
        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
        flattened_text_emb = self.flatten_emb_by_sentence(text_emb, text_len_mask)  # [num_words]

        candidate_starts, candidate_ends = spans(
            sentence_indices=flattened_sentence_indices,
            max_width=self.max_mention_width)
        candidate_starts.set_shape([None])
        candidate_ends.set_shape([None])

        candidate_mention_emb = self.get_mention_emb(flattened_text_emb, text_outputs, candidate_starts,
                                                     candidate_ends)  # [num_candidates, emb] WE CAN WRITE HERE GOLD STARTS AND
        candidate_mention_scores = self.get_mention_scores(candidate_mention_emb)  # [num_mentions, 1]
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [num_mentions]

        k = tf.to_int32(tf.floor(tf.to_float(tf.shape(text_outputs)[0]) * self.opt["mention_ratio"]))
        predicted_mention_indices = extract_mentions(candidate_mention_scores, candidate_starts,
                                                               candidate_ends, k)  # ([k], [k])
        predicted_mention_indices.set_shape([None])

        mention_starts = tf.gather(candidate_starts, predicted_mention_indices)  # [num_mentions]
        mention_ends = tf.gather(candidate_ends, predicted_mention_indices)  # [num_mentions]
        mention_emb = tf.gather(candidate_mention_emb, predicted_mention_indices)  # [num_mentions, emb]
        mention_scores = tf.gather(candidate_mention_scores, predicted_mention_indices)  # [num_mentions]

        mention_start_emb = tf.gather(text_outputs, mention_starts)  # [num_mentions, emb]
        mention_end_emb = tf.gather(text_outputs, mention_ends)  # [num_mentions, emb]
        mention_speaker_ids = tf.gather(speaker_ids, mention_starts)  # [num_mentions]

        max_antecedents = self.opt["max_antecedents"]
        antecedents, antecedent_labels, antecedents_len = get_antecedents(mention_starts, mention_ends,
                                                                                gold_starts, gold_ends, cluster_ids,
                                                                                max_antecedents)  # ([num_mentions, max_ant], [num_mentions, max_ant + 1], [num_mentions]
        antecedents.set_shape([None, None])
        antecedent_labels.set_shape([None, None])
        antecedents_len.set_shape([None])

        antecedent_scores = self.get_antecedent_scores(mention_emb, mention_scores, antecedents, antecedents_len,
                                                       mention_starts, mention_ends, mention_speaker_ids,
                                                       genre_emb)  # [num_mentions, max_ant + 1]

        loss = self.softmax_loss(antecedent_scores, antecedent_labels)  # [num_mentions]
        loss = tf.reduce_sum(loss)  # []

        return [candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, antecedents,
                antecedent_scores], loss

    def get_predicted_clusters(self, mention_starts, mention_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(mention_starts[predicted_index]), int(mention_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(mention_starts[i]), int(mention_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def init_from_saved(self):               
        log_dir = os.path.join(self.opt["log_root"], self.opt['name'])
        if isdir(log_dir):
            saver = tf.train.Saver()
            checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
            saver.restore(self.sess, checkpoint_path)
        else:
            print('{0} not found'.format(log_dir))

    def shutdown(self):
        tf.reset_default_graph()

    def save(self):
        log_dir = os.path.join(self.opt["log_root"])
        if isdir(log_dir):
            if isdir(os.path.join(log_dir, self.opt['name'])):  
                saver = tf.train.Saver()
                print('saving path ' + os.path.join(log_dir, self.opt['name'], 'model.max.ckpt'))
                saver.save(self.sess, os.path.join(log_dir, self.opt['name'], 'model.max.ckpt'))
            else:
                os.mkdir(self.opt['name'])
                saver = tf.train.Saver()
                print('saving path ' + os.path.join(log_dir, self.opt['name'], 'model.max.ckpt'))
                saver.save(self.sess, os.path.join(log_dir, self.opt['name'], 'model.max.ckpt'))
        else:
            os.mkdir(self.opt["log_root"])
            if isdir(os.path.join(log_dir, self.opt['name'])):  
                saver = tf.train.Saver()
                print('saving path ' + os.path.join(log_dir, self.opt['name'], 'model.max.ckpt'))
                saver.save(self.sess, os.path.join(log_dir, self.opt['name'], 'model.max.ckpt'))
            else:
                os.mkdir(self.opt['name'])
                saver = tf.train.Saver()
                print('saving path ' + os.path.join(log_dir, self.opt['name'], 'model.max.ckpt'))
                saver.save(self.sess, os.path.join(log_dir, self.opt['name'], 'model.max.ckpt'))

    def train(self, batch):
#        print(batch)
        self.start_enqueue_thread(batch, True)
        self.tf_loss, tf_global_step, _ = self.sess.run([self.loss, self.global_step, self.train_op])
        return self.tf_loss

    def predict(self, batch, out_file):
        self.start_enqueue_thread(batch, False)        
        candidate_starts, candidate_ends, mention_scores, mention_starts, mention_ends, antecedents, antecedent_scores = self.sess.run(self.predictions)
        
        
        
        _, _, _, _, _, _, gold_starts, gold_ends, _ = lambda batch: self.tensorize_example(batch, is_training=False)
        text_length = sum(len(s) for s in batch["sentences"])
        gold_spans = set(zip(gold_starts, gold_ends))

        if len(candidate_starts) > 0:
          sorted_starts, sorted_ends, _ = zip(*sorted(zip(candidate_starts, candidate_ends, mention_scores)))
        else:
          sorted_starts = []
          sorted_ends = []

        num_predictions = len(gold_spans)
        predicted_starts = sorted_starts[:num_predictions]
        predicted_ends = sorted_ends[:num_predictions]
        predicted_spans = set(zip(predicted_starts, predicted_ends))
        
        

        predicted_antecedents = self.get_predicted_antecedents(antecedents, antecedent_scores)

#        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(mention_starts, mention_ends,                                                                               predicted_antecedents)
        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(predicted_starts, predicted_ends,                predicted_antecedents)
        new_cluters = {}
        new_cluters[batch['doc_key']] = predicted_clusters
        outconll = utils.output_conll(out_file, batch, new_cluters)

        return outconll
