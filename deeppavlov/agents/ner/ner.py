# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import numpy as np
from parlai.core.agents import Agent

from . import config
from .dictionary import NERDictionaryAgent
from .ner_tagger import NERTagger
from .dictionary import get_char_dict
import re

CHAR_DICT = get_char_dict()


class NERAgent(Agent):
    """Named Entity Recognition agent"""

    @staticmethod
    def dictionary_class():
        """Define the dictionary agent"""
        return NERDictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Update command line arguments"""
        config.add_cmdline_args(argparser)
        NERAgent.dictionary_class().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        """Initialization of the agent"""
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
        """Observe the data from the teacher"""
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            dialogue = self.observation['text'].split(' ')[:-1]
            dialogue.extend(observation['text'].split(' '))
            observation['text'] = ' '.join(dialogue)
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        """Perform action on the observations"""
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        """Perform action on observations

        Args:
            observations: batch of observations

        Returns:
            batch_response: predicted tags for observatoins
        """
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch = self.batchify(observations)
        (x, xc), y = batch
        if 'labels' in observations[0]:

            self.loss = self.network.train_on_batch(x, xc, y)
            responses = self.network.predict(x, xc)
        else:
            responses = self.network.predict(x, xc)

        batch_response = [{'id': self.id} for _ in observations]

        for i in range(len(responses)):
            if responses[i] is not None:
                batch_response[i]['text'] = self.word_dict.labels_dict.vec2txt(responses[i])

        return batch_response

    def batchify(self, observations):
        """Create numpy ndarray from the given observations

        Args:
            observations: observations

        Returns:
            x - 2-D array of token indices
            xc - 3-D array of indiced of characters
            y - 2-D array of ground truth tag indices
        """
        batch_size = len(observations)
        x_list = []
        x_char_list = []
        y_list = []
        max_len = 0
        max_len_char = 0
        for observation in observations:
            if 'text' in observation:
                text = observation['text']
                # text = ' '.join(re.findall(r"[\w]+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]", text))
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
        # Handle the case of incomplete batch in the end of the dataset
        current_batch_size = len(x_list)
        x = np.ones([current_batch_size, max_len]) * self.word_dict[self.word_dict.null_token]
        xc = np.ones([current_batch_size, max_len, max_len_char]) * CHAR_DICT['<PAD>']
        y = np.ones([current_batch_size, max_len]) * self.word_dict.labels_dict[self.word_dict.labels_dict.null_token]

        for n, (x_item, x_char, y_item) in enumerate(zip(x_list, x_char_list, y_list)):
            n_tokens = len(x_item)
            x[n, :n_tokens] = x_item
            y[n, :n_tokens] = y_item
            for k, characters in enumerate(x_char):
                xc[n, k, :len(characters)] = characters
        return (x, xc), y

    def save(self, fname=None):
        """Save the parameters of the agent to a file"""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            try:
                self.network.save(fname)
            except BaseException:
                print('[ WARN: Saving failed... continuing anyway. ]')

    def load(self, fname=None):
        """Load the parameters of the agent from the file"""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            try:
                self.network.load(fname)
            except BaseException:
                print('[ WARN: Saving failed... continuing anyway. ]')

    def shutdown(self):
        """Reset the model"""
        # Reset the model
        if not self.is_shared:
            self.network.shutdown()
