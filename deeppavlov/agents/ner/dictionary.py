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
import os
import pickle
from collections import defaultdict
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str


def get_char_dict():
    """Create character dict from predefined set of characters

    Returns:
        char_dict: dictionary of characters

    """
    base_characters = u'\"#$%&\'()+,-./0123456789:;<>?ABCDEFGHI' \
                      u'JKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstu' \
                      u'vwxyz«\xad»×АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪ' \
                      u'ЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё‑–' \
                      u'—“”€№…’'
    characters = ['<UNK>'] + ['<PAD>'] + list(base_characters)
    char_dict = defaultdict(int)
    for i, ch in enumerate(characters):
        char_dict[ch] = i

    return char_dict


class NERDictionaryAgent(DictionaryAgent):
    """Named Entity Recognition dictionary agent"""

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command line arguments"""
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--dict_class', default=class2str(NERDictionaryAgent),
            help='Sets the dictionary\'s class'
        )

    def __init__(self, opt, shared=None):
        """Initialize NER dictionary agent"""
        child_opt = copy.deepcopy(opt)
        # child_opt['model_file'] += '.labels'
        child_opt['dict_file'] = child_opt['dict_file'] + '.labels.dict'
        self.labels_dict = DictionaryAgent(child_opt, shared)
        self.char_dict = get_char_dict()
        super().__init__(opt, shared)

    def observe(self, observation):
        """Get the data from the observation"""
        observation = copy.deepcopy(observation)
        labels_observation = copy.deepcopy(observation)
        labels_observation['text'] = None
        observation['labels'] = None
        self.labels_dict.observe(labels_observation)
        return super().observe(observation)

    def act(self):
        self.labels_dict.act()
        super().act()
        return {'id': 'NERDictionary'}

    def save(self, filename=None, append=False, sort=True):
        """Save dictionary to the file

        Args:
            filename: filename of the dictionary
            append: boolean whether to append to the existing dict
            sort: boolean which determines whether to sort the dict or not

        Returns:
            None
        """
        filename = self.opt['model_file'] if filename is None else filename
        self.labels_dict.save(filename + '.labels.dict')
        return super().save(filename, append, sort)

    def tokenize(self, text, building=False):
        """Tokenize given text"""
        return text.split(' ') if text else []


