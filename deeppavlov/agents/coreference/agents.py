"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
from parlai.core.agents import Agent
from . import config
from .models import CorefModel
from . import utils

class CoreferenceAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        self.id = 'Coreference_Agent'
        self.episode_done = True
        super().__init__(opt, shared)

        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        self.obs_dict = None
        self.iterations = 0
        self.rep_iter = opt['rep_iter']
        self.model = CorefModel(opt)
        if self.opt['pretrained_model']:
            print('[ Initializing model from checkpoint ]')
            self.model.init_from_saved()
        else:
            print('[ Initializing model from scratch ]')

    def observe(self, observation):
        self.observation = copy.deepcopy(observation)
<<<<<<< HEAD
        self.obs_dict = utils.conll2modeldata(self.observation)
        return self.obs_dict
=======
        self.obs_dict = utils.conll2modeldata(observation)
#        return self.obs_dict
>>>>>>> 7f6eb564c90900aa9994ae9969039c5965183c97

    def act(self):
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")
        if self.observation['mode'] == 'train':
            self.tf_loss = self.model.train(self.obs_dict)
            if self.observation['iter_id'] % self.rep_iter == 0:
                self.iterations += self.rep_iter
                print('iter: {} | Loss: {}'.format(self.iterations, self.tf_loss))
            act_dict = {'iter_id': self.observation['iter_id'], 'Loss': self.tf_loss}
            act_dict['id'] = self.id
            act_dict['epoch_done'] = self.observation['epoch_done']
            act_dict['mode'] = self.observation['mode']
            act_dict['conll'] = False
            act_dict['loss'] = self.tf_loss
            return act_dict
        elif self.observation['mode'] == 'valid':
            # tf_loss = self.model.train_op(observation)
            conll = self.model.predict(self.obs_dict, self.observation)
            conll['conll'] = True
            conll['iter_id'] = self.observation['iter_id']
            return conll
        elif self.observation['mode'] == 'test':
            conll = self.model.predict(self.obs_dict, self.observation)
            conll['conll'] = True
            conll['iter_id'] = self.observation['iter_id']
            return conll

    def eval(self):
        loss = self.model.eval(self.obs_dict)
        return loss

    def predict(self):
        y = self.model.predict(self.obs_dict, self.observation)
        return y

    def save(self):
        self.model.save()

    def load(self):
        self.model.init_from_saved()

    def shutdown(self):
        if not self.is_shared:
            if self.model is not None:
                self.model.shutdown()
            self.model = None

    def report(self):
        return {'loss': self.tf_loss}