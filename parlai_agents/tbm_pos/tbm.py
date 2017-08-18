import copy

import os
import torch
from parlai.core.agents import Agent

from .pos_tagger import POSTagger
from .dictionary import POSDictionaryAgent
from .trainer import TeacherForcingTrainer, BeamTrainer
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

        self.network = POSTagger(opt, self.word_dict)

        state_dict = None
        if opt.get('model_file') and os.path.isfile(opt['model_file']):
            data = torch.load(opt['model_file'])
            state_dict = data['state_dict']
        elif opt.get('pretrained_model'):
            data = torch.load(opt['pretrained_model'])
            state_dict = data['state_dict']

        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        if opt['trainer_type'] == 'naive':
            self.model = TeacherForcingTrainer(opt['learning_rate'], self.network)
        elif opt['trainer_type'] == 'beam':
            self.model = BeamTrainer(opt['learning_rate'], opt['beam'], self.network)
        else:
            raise RuntimeError('not applicable model type')

        super().__init__(opt, shared)

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

            params = {
                'state_dict': {
                    'network': self.network.state_dict()
                    # 'optimizer': self.model.state_dict()
                },
                'word_dict': self.word_dict,
                'config': self.opt,
            }
            try:
                torch.save(params, fname)
            except BaseException:
                print('[ WARN: Saving failed... continuing anyway. ]')

    # def report(self):
    #     if self.loss is not None:
    #         return '[train] train loss = %.2f' % self.loss.data[0]
    #     else:
    #         return '[train] Nothing to report'
