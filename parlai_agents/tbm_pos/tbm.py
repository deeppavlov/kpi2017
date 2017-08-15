import copy

import os
import torch
from parlai.core.agents import Agent

from .dictionary import POSDictionaryAgent
from .model import MachineTrainer
from . import config


class NaiveAgent(Agent):

    @staticmethod
    def dictionary_class():
        return POSDictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        NaiveAgent.dictionary_class().add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        self.id = 'NaiveTBMAgent'
        self.episode_done = True
        self.loss = None

        # Only create an empty dummy class when sharing
        if shared is not None:
            self.is_shared = True
            return
        self.is_shared = False
        self.word_dict = NaiveAgent.dictionary_class()(opt)
        if opt.get('model_file') and os.path.isfile(opt['model_file']):
            self._init_from_saved(opt, opt['model_file'])
        else:
            if opt.get('pretrained_model'):
                self._init_from_saved(opt, opt['pretrained_model'])
            else:
                self._init_from_scratch(opt)
        super().__init__(opt, shared)

    def _init_from_saved(self, opt, fname):
        data = torch.load(fname)
        self.model = MachineTrainer(opt, self.word_dict, data['state_dict'])

    def _init_from_scratch(self, opt):
        self.model = MachineTrainer(opt, self.word_dict)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if 'text' in observation:
            observation['text'] = observation['text'].lower()
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

        if 'labels' in observations[0]:
            self.loss, paths = self.model.train(batch)
        else:
            paths = self.model.forward(batch)

        batch_response = [{
            'id': self.id,
            'text': self.word_dict.labels_dict.vec2txt(path) if path else None
        } for path in paths]

        return batch_response

    def batchify(self, observations):
        batch = []
        for observation in observations:
            words = observation.get('text', None)
            tags = self.word_dict.labels_dict.txt2vec(observation['labels'][0]) if 'labels' in observation else None
            batch.append((words, tags))
        return batch

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save(fname)

    # def report(self):
    #     if self.loss is not None:
    #         return '[train] train loss = %.2f' % self.loss.data[0]
    #     else:
    #         return '[train] Nothing to report'
