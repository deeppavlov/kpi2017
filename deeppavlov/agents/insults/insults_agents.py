import copy
import os
import numpy as np

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str
from scipy.io import mmwrite, mmread

from . import config
from .model import InsultsModel
from .utils import create_vectorizer_selector, get_vectorizer_selector, vectorize_select_from_data


class EnsembleInsultsAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        ensemble = argparser.add_argument_group('Ensemble parameters')
        ensemble.add_argument('--model_files', type=str, default=None, nargs='+',
                              help='list of all the model files for the ensemble')
        ensemble.add_argument('--model_names', type=str, default=None, nargs='+',
                              help='list of all the model names for the ensemble')
        ensemble.add_argument('--model_coefs', type=str, default=None, nargs='+',
                              help='list of all the model coefs for the ensemble')
    
    def __init__(self, opt, shared=None):
        self.id = 'InsultsAgent'
        self.episode_done = True
        super().__init__(opt, shared)
        if shared is not None:
            self.is_shared = True
            return
        # Set up params/logging/dicts
        self.is_shared = False

        self.models = []
        for i, model_name in enumerate(opt.get('model_names', [])):
            print('Model:', model_name)
            model_file = opt.get('model_files', [])[i]
            opt['pretrained_model'] = model_file
            print('Model file:', model_file)
            if model_name == 'cnn_word':
                self.word_dict = None
                ## NO EMBEDDINGS NOW
                self.embedding_matrix = None
                self.num_ngrams = None
            if model_name == 'log_reg' or model_name == 'svc':
                self.word_dict = None
                self.embedding_matrix = None
                self.num_ngrams = 6
            self.models.append(InsultsModel(model_name, self.word_dict, self.embedding_matrix, opt))
            if model_name == 'log_reg' or model_name == 'svc':
                print('Reading vectorizers and selectors')
                self.models[i].vectorizers, self.models[i].selectors = get_vectorizer_selector(model_file,
                                                                                   self.num_ngrams)
        self.model_coefs = [float(coef) for coef in opt.get('model_coefs', [])]
        print('model coefs:', self.model_coefs)
        self.n_examples = 0

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

    def _predictions2text(self, predictions):
        y = ['Insult' if ex > 0.5 else 'Non-insult' for ex in predictions]
        return y

    def _text2predictions(self, predictions):
        y = [1 if ex == 'Insult' else 0 for ex in predictions]
        return y

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

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):

        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]
        predictions = [[] for _ in range(batch_size)]
        for j, model in enumerate(self.models):
            examples = [model._build_ex(obs) for obs in observations]
            valid_inds = [i for i in range(batch_size) if examples[i] is not None]
            examples = [ex for ex in examples if ex is not None]

            batch = model._batchify(examples, self.word_dict)
            prediction = model.predict(batch)
            for i in range(len(prediction)):
                predictions[valid_inds[i]].append(prediction[i])

        for i in range(batch_size):
            if len(predictions[i]):
                prediction = self.weighted_sum(predictions[i])
                batch_reply[i]['text'] = self._predictions2text([prediction])[0]
                batch_reply[i]['score'] = prediction

        return batch_reply

    def weighted_sum(self, predictions):
        result = 0
        for j in range(len(predictions)):
            result += self.model_coefs[j] * predictions[j]
        result = result / sum(self.model_coefs)
        return result


class InsultsAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)

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

        if (self.model.from_saved == True ):
            print ('Reading vectorizers and selectors')
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
            batch = self.model._batchify(examples)
            predictions = self.model.update(batch)
            predictions = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]
        else:
            batch = self.model._batchify(examples)
            predictions = self.model.predict(batch)
            predictions_text = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions_text[i]
                batch_reply[valid_inds[i]]['score'] = predictions[i]

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


    def _predictions2text(self, predictions):
        y = ['Insult' if ex > 0.5 else 'Non-insult' for ex in predictions]
        return y

    def _text2predictions(self, predictions):
        y = [1 if ex == 'Insult' else 0 for ex in predictions]
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
        self.observations_ = []

    def batch_act(self, observations):
        self.observations_ += observations

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
            batch = self.model._batchify(examples)
            predictions = self.model.predict(batch).reshape(-1)
            predictions_text = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions_text[i]
                batch_reply[valid_inds[i]]['score'] = predictions[i]
            #comment = 'shut the fuck up .. you and the rest of your faggot friend should be burned at the stake'
            #print('Predicted for comment', self.model.predict([comment]))
        return batch_reply

    def save(self):
        if not self.is_shared:
            train_data = [observation['text'] for observation in self.observations_ if 'text' in observation.keys()]
            train_labels = self._text2predictions([observation['labels'][0] for observation in self.observations_ if 'labels' in observation.keys()])

            #with open(self.opt['model_file'] + '_using_train.txt', 'w') as f:
            #    for comment in train_data:
            #        f.write(comment + '\n')


            self.model.num_ngrams = self.num_ngrams

            if self.model.from_saved == False:
                print('Creating vectorizers and selectors')
                create_vectorizer_selector(train_data, train_labels, self.opt['model_file'],
                                           ngram_list=[1, 2, 3, 4, 5, 3],
                                           max_num_features_list=[2000, 4000, 100, 1000, 1000, 2000],
                                           analyzer_type_list=['word', 'word', 'word', 'char', 'char', 'char'])
                print('Reading vectorizers and selectors')
                self.model.vectorizers, self.model.selectors = get_vectorizer_selector(self.opt['model_file'],  self.num_ngrams)

                print('Training model', self.model_name)
                self.model.update([train_data, train_labels])
                print('\n[model] trained loss = %.4f | acc = %.4f | auc = %.4f' %
                      (self.model.train_loss, self.model.train_acc, self.model.train_auc,))
                self.model.save()



                return

        comment = 'shut the fuck up .. you and the rest of your faggot friend should be burned at the stake'
        print('Predicted for comment', self.model.predict([comment]))

        self.model.save()
        return







