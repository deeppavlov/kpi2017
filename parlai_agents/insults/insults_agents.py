import copy
import numpy as np
from . import config
from .model import InsultsModel

from keras.preprocessing.sequence import pad_sequences

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str



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

        print('create word dict')
        self.word_dict = InsultsAgent.dictionary_class()(opt)
        ## NO EMBEDDINGS NOW
        #print('create embedding matrix')
        #self.embedding_matrix = load_embeddings(opt, self.word_dict.tok2ind)
        self.embedding_matrix = None
        print('create model')
        self.models = []
        for model_name in (opt['models'].split(' ')):
            self.models.append(InsultsModel(model_name, self.word_dict, self.embedding_matrix, opt))
        self.models_coefs = [1.]
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
        batch = self._batchify(examples)

        if 'labels' in observations[0]:
            self.n_examples += len(examples)
            for i in range(len(self.models)):
                self.models[i].update(batch)
        else:
            list_predictions = []
            for i in range(len(self.models)):
                if self.models[i].model_type == 'nn':
                    list_predictions.append(self.models[i].predict(batch))
                if self.models[i].model_type == 'ngrams':
                    list_predictions.append(self.models[i].predict(batch).reshape(-1))
            # weighted sum of predictions
            predictions = np.array(list_predictions).T.dot(np.array(self.models_coefs))
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
        question = []
        for ex in batch:
            question.append(self.word_dict.txt2vec(ex['question']))
        question = pad_sequences(question, maxlen=self.opt['max_sequence_length'], padding='post')
        if len(batch[0]) == 2:
            y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
            return question, y
        else:
            return question

    def _predictions2text(self, predictions):
        y = ['Insult' if ex > 0.5 else 'Non-insult' for ex in predictions]
        return y

    def report(self):
        info = ''
        args = ()
        for i in range(len(self.models)):
            info += '\n[model %d] updates = %d | exs = %d | loss = %.4f | acc = %.4f | auc = %.4f'
            args += (i, self.models[i].updates, self.n_examples,
                     self.models[i].train_loss, self.models[i].train_acc, self.models[i].train_auc,)
        return (info % args)
        #return (
        #    '[train] updates = %d | exs = %d | loss = %.4f | acc = %.4f | auc = %.4f'%
        #    (self.model.updates, self.n_examples,
        #     self.model.train_loss, self.model.train_acc, self.model.train_auc))

