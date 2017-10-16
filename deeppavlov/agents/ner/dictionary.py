import copy
import os
import pickle
from collections import defaultdict
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str


def get_char_dict():
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

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--dict_class', default=class2str(NERDictionaryAgent),
            help='Sets the dictionary\'s class'
        )

    def __init__(self, opt, shared=None):
        child_opt = copy.deepcopy(opt)
        # child_opt['model_file'] += '.labels'
        child_opt['dict_file'] = child_opt['dict_file'] + '.labels.dict'
        self.labels_dict = DictionaryAgent(child_opt, shared)
        self.char_dict = get_char_dict()
        super().__init__(opt, shared)

    def observe(self, observation):
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
        filename = self.opt['model_file'] if filename is None else filename
        self.labels_dict.save(filename + '.labels.dict')
        return super().save(filename, append, sort)

    def tokenize(self, text, building=False):
        return text.split(' ') if text else []


