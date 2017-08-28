import copy
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str
from scipy.io import mmwrite, mmread

from . import config
from .model import InsultsModel
from .utils import create_vectorizer_selector, get_vectorizer_selector, vectorize_select_from_data


class InsultsDictionaryAgent(DictionaryAgent):

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--dict_class', default=class2str(InsultsDictionaryAgent)
        )

    def act(self):
        """Add only words passed in the 'text' field of the observation to this dictionary."""
        text = self.observation.get('text')
        if text:
            self.add_to_dict(self.tokenize(text))
        return {'id': 'InsultsDictionary'}

class InsultsAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        InsultsAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return InsultsDictionaryAgent

    def __init__(self, opt, shared=None):
        self.id = 'InsultsAgent'
        self.episode_done = True
        super().__init__(opt, shared)
        if shared is not None:
            self.is_shared = True
            return
        # Set up params/logging/dicts
        self.is_shared = False

        self.model_name = opt['model_name']

        if self.model_name == 'cnn_word':
            print('create word dict')
            self.word_dict = InsultsAgent.dictionary_class()(opt)
            ## NO EMBEDDINGS NOW
            #print('create embedding matrix')
            #self.embedding_matrix = load_embeddings(opt, self.word_dict.tok2ind)
            self.embedding_matrix = None
            self.num_ngrams = None
        if self.model_name == 'log_reg' or self.model_name == 'svc':
            self.word_dict = None
            self.embedding_matrix = None
            self.num_ngrams = 6

        print('create model', self.model_name)
        self.model = InsultsModel(self.model_name, self.word_dict, self.embedding_matrix, opt)
        self.n_examples = 0

        if (self.model.model_type == 'ngrams' and
                (os.path.isfile(self.opt['model_file'] + '_ngrams_vect_special.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_0.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_1.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_2.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_3.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_4.bin') and
                     os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_5.bin')) ):
            print ('reading vectorizers selectors')
            self.model.vectorizers, self.model.selectors = get_vectorizer_selector(self.opt['model_file'],  self.num_ngrams)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]
        examples = [self._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batch_size) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]

        if 'labels' in observations[0]:
            self.n_examples += len(examples)
            batch = self._batchify(examples)
            predictions = self.model.update(batch)
            predictions = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]
        else:
            batch = self._batchify(examples)
            predictions = self.model.predict(batch)
            predictions = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]

        return batch_reply

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

    def _batchify(self, batch):
        if self.model.model_type == 'nn':
            question = []
            for ex in batch:
                question.append(self.word_dict.txt2vec(ex['question']))
            question = pad_sequences(question, maxlen=self.model.opt['max_sequence_length'], padding='post')
            if len(batch[0]) == 2:
                y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
                return question, y
            else:
                return question

        if self.model.model_type == 'ngrams':
            questions = []
            for ex in batch:
                questions.append(ex['question'])

            ngrams_questions = vectorize_select_from_data(questions, self.model.vectorizers, self.model.selectors)
            if len(batch[0]) == 2:
                y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
                return ngrams_questions, y
            else:
                return ngrams_questions

    def _predictions2text(self, predictions):
        y = ['Insult' if ex > 0.5 else 'Non-insult' for ex in predictions]
        return y

    def _text2predictions(self, predictions):
        y = [1. if ex == 'Insult' else 0 for ex in predictions]
        return y

    def report(self):
        report = dict()
        report['updates'] = self.model.updates
        report['n_examples'] = self.n_examples
        report['loss'] = self.model.train_loss
        report['accuracy'] = self.model.train_acc
        report['auc'] = self.model.train_auc
        return report

    def save(self):
        self.model.save()


class OneEpochAgent(InsultsAgent):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.observation = ''
        self.observations = []

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        try:
            prev_dialogue = self.observation['text']
            prev_labels = self.observation['labels']
            observation['text'] = prev_dialogue + '\n' + observation['text']
            observation['labels'] = prev_labels + observation['labels']
            self.observation = observation
        except TypeError:
            self.observation = observation
        except KeyError:
            self.observation = observation
        return observation

    def batch_act(self, observations):
        self.observations += observations

        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]
        examples = [self._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batch_size) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]

        if 'labels' in observations[0]:
            self.n_examples += len(examples)
        else:
            batch = self._batchify(examples)
            predictions = self.model.predict(batch).reshape(-1)
            predictions = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]
            if 1:
                comment = 'your absolutely right , there i no way of explain it to a moron like you - go back to smokiung your crack , 8ss-hole ?! ?!'
                print('Comment:', comment)
                comment_features = vectorize_select_from_data([comment], self.model.vectorizers, self.model.selectors)
                print('Features:', comment_features.tocsr())
                print('Predicted:', (self.model.predict(comment_features.tocsr()[0:1,:])))
        return batch_reply

    def save(self):
        if not self.is_shared:
            train_data = [observation['text'] for observation in self.observations if 'text' in observation.keys()]
            train_labels = self._text2predictions([observation['labels'][0] for observation in self.observations if 'labels' in observation.keys()])

            # this should be done once for each launch!!!
            if not (os.path.isfile(self.opt['model_file'] + '_ngrams_vect_special.bin') and
                    os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_0.bin') and
                    os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_1.bin') and
                    os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_2.bin') and
                    os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_3.bin') and
                    os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_4.bin') and
                    os.path.isfile(self.opt['model_file'] + '_ngrams_vect_general_5.bin') ):
                print ('No vectorized data found. Vectorizing train data')
                create_vectorizer_selector(train_data, train_labels, self.opt['model_file'],
                                           ngram_list=[1, 2, 3, 4, 5, 3],
                                           max_num_features_list=[2000, 4000, 100, 1000, 1000, 2000],
                                           analyzer_type_list=['word', 'word', 'word', 'char', 'char', 'char'])
            if self.model.vectorizers is None:
                print ('Get vectorizers and selectors')
                self.model.vectorizers, self.model.selectors = get_vectorizer_selector(self.opt['model_file'],  self.num_ngrams)
            self.model.num_ngrams = self.num_ngrams

            if os.path.isfile(self.opt['model_file'] + '_train_vectorized.mtx'):
                print('Reading vectorized train dataset')
                X_train = mmread(self.opt['model_file'] + '_train_vectorized.mtx')
            else:
                print('Vectorizing train dataset')
                X_train = vectorize_select_from_data(train_data, self.model.vectorizers, self.model.selectors)
                mmwrite(self.opt['model_file'] + '_train_vectorized', X_train)

            print('Training model', self.model_name)
            self.model.update([X_train, train_labels])

            if 1:
                comment = 'your absolutely right , there i no way of explain it to a moron like you - go back to smokiung your crack , 8ss-hole ?! ?!'#'haha fuck you bitch your a fukn cripled cunt fuck you motherfucker stay off thi page unles you want to die bitch ?! ?!'
                print('Comment:', comment)
                comment_features = vectorize_select_from_data([comment], self.model.vectorizers, self.model.selectors)
                print('Features:', comment_features.tocsr())
                print('Predicted:', (self.model.predict(comment_features.tocsr()[0:1,:])))

        print ('\n[model] trained loss = %.4f | acc = %.4f | auc = %.4f' %
               (self.model.train_loss, self.model.train_acc, self.model.train_auc,))
        self.model.save()







