import copy

import numpy as np
from parlai.core.agents import Agent

from . import config
from .dictionary import NERDictionaryAgent
from .ner_tagger import NERTagger
from .dictionary import char_dict

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

    def get_default_char_dict(self):
        char_dict = dict()

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
        (x, xc), y = batch
        if 'labels' in observations[0]:
            self.loss = self.network.train_on_batch(x, xc, y)
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
        x_char_list = []
        y_list = []
        max_len = 0
        max_len_char = 0
        for observation in observations:
            text = observation.get('text', None)

            current_char_list = []
            tokens = text.split()
            for token in tokens:
                characters = [self.word_dict.char_dict[ch] for ch in token]
                current_char_list.append(characters)
                max_len_char = max(max_len_char, len(token))
            x_char_list.append(current_char_list)

            tokens = self.word_dict.txt2vec(text)
            tags = self.word_dict.labels_dict.txt2vec(observation['labels'][0]) if 'labels' in observation else None
            max_len = max(len(tokens), max_len)
            x_list.append(tokens)
            y_list.append(tags)
        x = np.ones([batch_size, max_len]) * self.word_dict[self.word_dict.null_token]
        xc = np.ones([batch_size, max_len, max_len_char]) * char_dict['<PAD>']
        y = np.ones([batch_size, max_len]) * self.word_dict.labels_dict[self.word_dict.labels_dict.null_token]

        for n, (x_item, x_char, y_item) in enumerate(zip(x_list, x_char_list, y_list)):
            n_tokens = len(x_item)
            x[n, :n_tokens] = x_item
            y[n, :n_tokens] = y_item
            for k, characters in enumerate(x_char):
                xc[n, k, :len(characters)] = characters
        return (x, xc), y

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


