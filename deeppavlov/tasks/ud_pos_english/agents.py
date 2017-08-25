from parlai.core.agents import Agent
from parlai.core.dialog_teacher import DialogTeacher
from .build import build
from .conllu_parser import read_conllu_file, get_pos_tags

import random

import json
import os


def _path(opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]

    if dt != 'train':
        dt = 'dev'
    # elif dt != 'train' and dt != 'test':
    #     raise RuntimeError('Not valid datatype.')

    prefix = os.path.join(opt['datapath'], 'UD')
    path = os.path.join(prefix, 'en-ud-' + dt + '.conllu')
    # stats_path = os.path.join(prefix, 'stats.xml')

    return path


class DefaultTeacher(DialogTeacher):

    def observe(self, observation):
        """Process observation for metrics. """
        if self.lastY is not None:
            if observation.get('text'):
                ys = self.lastY[0].split(' ')
                xs = [{'text': x} for x in observation['text'].split(' ')]
                for x, y in zip(xs, ys):
                    self.metrics.update(x, (y, ))
            else:
                self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def __init__(self, opt, shared=None):
        self.id = 'tbm'
        self.datatype = opt['datatype']
        data_path = _path(opt)
        opt['datafile'] = data_path

        super().__init__(opt, shared)

    def setup_data(self, path):
        pos = list(get_pos_tags(read_conllu_file(path)))
        for sentence in pos:
            new_episode = True
            words, tags = zip(*sentence)
            yield (' '.join(words), [' '.join(tags)], '1', None), new_episode


class WordsTeacher(DefaultTeacher):
    # all possible answers for the questions
    cands = {'START', 'ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ', 'X', 'NOUN', 'DET', 'PROPN', 'NUM',
             'VERB', 'PART', 'PRON', 'SCONJ'}

    def label_candidates(self):
        return self.cands

    def setup_data(self, path):
        print('loading: ' + path)
        pos = list(get_pos_tags(read_conllu_file(path)))

        for sentence in pos:
            new_episode = True
            for word, pos_tag in sentence:
                yield (word, [pos_tag], None, None, None), new_episode
                new_episode = False
