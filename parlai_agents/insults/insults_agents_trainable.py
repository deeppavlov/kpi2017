import copy
import numpy as np
from . import config
from .model import InsultsModel

from keras.preprocessing.sequence import pad_sequences

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str
from keras import backend as K


class InsultsAgentTrainable(Agent):

    def __init__(self, model_name, word_dict, opt, shared=None):
        self.id = 'InsultsAgentTrainable'
        self.episode_done = True
        super().__init__(opt, shared)
        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False

        self.word_dict = word_dict
        ## NO EMBEDDINGS NOW
        #print('create embedding matrix')
        #self.embedding_matrix = load_embeddings(opt, self.word_dict.tok2ind)
        self.embedding_matrix = None
        print('create model')
        self.model = InsultsModel(model_name, self.word_dict, self.embedding_matrix, opt)
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
            self.model.update(batch)
        else:
            if self.model.model_type == 'nn':
                predictions = self.model.predict(batch).reshape(-1)
            if self.model.model_type == 'ngrams':
                predictions = self.model.predict(batch).reshape(-1)
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
        info += '\n[model %d] updates = %d | exs = %d | loss = %.4f | acc = %.4f | auc = %.4f'
        args += (i, self.model.updates, self.n_examples,
                 self.model.train_loss, self.model.train_acc, self.model.train_auc,)
        print (args)
        return (info % args)

    def save(self):
        self.model.save()




