import copy
import os

from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str


class POSDictionaryAgent(DictionaryAgent):

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--dict_class', default=class2str(POSDictionaryAgent),
            help='Sets the dictionary\'s class'
        )

    def __init__(self, opt, shared=None):
        child_opt = copy.deepcopy(opt)
        # child_opt['model_file'] += '.labels'
        child_opt['dict_file'] = os.path.splitext(child_opt['dict_file'])[0] + '.labels.dict'
        self.labels_dict = DictionaryAgent(child_opt, shared)
        super().__init__(opt, shared)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if 'text' in observation:
            observation['text'] = observation['text'].lower()
        labels_observation = copy.deepcopy(observation)
        labels_observation['text'] = None
        observation['labels'] = None
        self.labels_dict.observe(labels_observation)
        return super().observe(observation)

    def act(self):
        self.labels_dict.act()
        super().act()
        return {'id': 'POSDictionary'}

    def save(self, filename=None, append=False, sort=True):
        filename = self.opt['model_file'] if filename is None else filename
        self.labels_dict.save(os.path.splitext(filename)[0] + '.labels.dict')
        return super().save(filename, append, sort)

    def tokenize(self, text, building=False):
        return text.split(' ')
