import copy

import numpy as np
from parlai.core.agents import Agent

from . import config
from .dictionary import NERDictionaryAgent
from .ner_tagger import NERTagger


class NERAgent(Agent):

    @staticmethod
    def dictionary_class():
        return NERDictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        NERAgent.dictionary_class().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        self.id = 'NERAgent'
        self.episode_done = True
        self.loss = None

        # Only create an empty dummy class when sharing
        if shared is not None:
            self.is_shared = True
            return
        self.is_shared = False
        self.word_dict = NERAgent.dictionary_class()(opt)
        self.network = NERTagger(opt, self.word_dict)

        super().__init__(opt, shared)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            dialogue = self.observation['text'].split(' ')[:-1]
            dialogue.extend(observation['text'].split(' '))
            observation['text'] = ' '.join(dialogue)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):

        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch = self.batchify(observations)
        x, y = batch
        if 'labels' in observations[0]:
            self.loss = self.network.train_on_batch(x, y)
            responses = [None for _ in range(len(x))]
        else:
            responses = self.network.predict(x)

        batch_response = [{
            'id': self.id,
            'text': self.word_dict.labels_dict.vec2txt(response) if response is not None else None
        } for response in responses]

        return batch_response

    def batchify(self, observations):
        batch_size = len(observations)
        x_list = []
        y_list = []
        max_len = 0
        for observation in observations:
            text = observation.get('text', None)
            tokens = self.word_dict.txt2vec(text)
            tags = self.word_dict.labels_dict.txt2vec(observation['labels'][0]) if 'labels' in observation else None
            max_len = max(len(tokens), max_len)
            x_list.append(tokens)
            y_list.append(tags)
        x = np.ones([batch_size, max_len]) * self.word_dict[self.word_dict.null_token]
        y = np.ones([batch_size, max_len]) * self.word_dict.labels_dict[self.word_dict.labels_dict.null_token]
        for n, (x_item, y_item) in enumerate(zip(x_list, y_list)):
            n_tokens = len(x_item)
            x[n, :n_tokens] = x_item
            y[n, :n_tokens] = y_item
        return x, y

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")

            try:
                self.network.save(fname)
            except BaseException:
                print('[ WARN: Saving failed... continuing anyway. ]')

    def shutdown(self):
        if not self.is_shared:
            self.network.shutdown()
