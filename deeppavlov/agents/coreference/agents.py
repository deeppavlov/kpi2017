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
import time
import tensorflow as tf
from parlai.core.agents import Agent
from . import config
from .models import CorefModel
from . import utils
import parlai.core.build_data as build_data
from os.path import join, isdir

def bdfa(opt):
    
    embed_url = '0B7A8-2DSIVoeelVIT1BMUFVLSnM'
    vocab_url = '0B7A8-2DSIVoed0FIMXdqU0FlQ3M'
    # get path to data directory and create folders tree
    dpath = join(opt['model_file'])
    # define languages
    language = opt['language']
    dpath = join(dpath, language, 'agent')
    build_data.make_dir(dpath)
    
    if not isdir(join(dpath, 'embeddings')):
        build_data.make_dir(join(dpath, 'embeddings'))
        print('[Download the word embeddings]...')
        build_data.download_from_google_drive(embed_url, join(dpath, 'embeddings', 'embeddings_lenta_100.vec'))
        print('[End of download the word embeddings]...')
    
    if not isdir(join(dpath, 'vocab')):
        build_data.make_dir(join(dpath, 'vocab'))
        print('[Download the chars vocalibary]...')
        build_data.download_from_google_drive(vocab_url, join(dpath, 'vocab', 'char_vocab.{}.txt'.format(language)))
        print('[End of download the chars vocalibary]...')
    
    if not isdir(join(dpath, 'logs', opt['name'])):
        build_data.make_dir(join(dpath, 'logs', opt['name']))
        
    if not isdir(join(dpath, 'reports')):
        build_data.make_dir(join(dpath, 'reports', 'response_files'))
        build_data.make_dir(join(dpath, 'reports', 'results'))
        build_data.make_dir(join(dpath, 'reports', 'predictions'))
    return None 

class CoreferenceAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        
        bdfa(opt)
        
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
        self.start = None
        self.tf_loss = None
        self.rep_iter = opt['rep_iter']
        self.nitr = opt['nitr']
        self.model = CorefModel(opt)
        
        self.saver = tf.train.Saver()
        if self.opt['pretrained_model']:
            print('[ Initializing model from checkpoint ]')
            self.model.init_from_saved(self.saver)
        else:
            print('[ Initializing model from scratch ]')

    def observe(self, observation):
        print('agent_obs {}'.format(0))
        self.observation = copy.deepcopy(observation)
        self.obs_dict = utils.conll2modeldata(self.observation)
        return self.obs_dict

    def act(self):
        print('agent_act {}'.format(0))
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")
        if self.observation['mode'] == 'train':
            self.tf_loss = self.model.train(self.obs_dict)
            self.remaining_time()
            act_dict = {'iter_id': self.observation['iter_id'], 'Loss': self.tf_loss}
            act_dict['id'] = self.id
            act_dict['epoch_done'] = self.observation['epoch_done']
            act_dict['mode'] = self.observation['mode']
            act_dict['conll'] = False
            act_dict['loss'] = self.tf_loss
            act_dict['iteration'] = self.iterations
            return act_dict
        elif self.observation['mode'] == 'valid':
            conll = dict()
            conll_str = self.model.predict(self.obs_dict, self.observation)
            conll['conll'] = True
            conll['iter_id'] = self.observation['iter_id']
            conll['conll_str'] = conll_str
            return conll
        elif self.observation['mode'] == 'test':
            conll = dict()
            conll_str = self.model.predict(self.obs_dict, self.observation)
            conll['conll'] = True
            conll['iter_id'] = self.observation['iter_id']
            conll['conll_str'] = conll_str
            return conll

    def predict(self):
        y = self.model.predict(self.obs_dict, self.observation)
        return y
    
    def prediction(self, path):
        y = self.model.predict(self.obs_dict, self.observation)
        utils.dict2conll(y, path)
        return None

    def save(self):
        self.model.save(self.saver)

    def shutdown(self):
        if not self.is_shared:
            if self.model is not None:
                self.model.shutdown()
            self.model = None

    def report(self):
        return {'loss': self.tf_loss}
    
    def remaining_time(self):
        if self.observation['iter_id'] % self.rep_iter == 0:
            self.iterations += self.rep_iter
            n = self.nitr*100 - self.iterations
            t = time.time() - self.start
            r_time = n*(t/self.rep_iter)
            hours = int(r_time/(60**2))
            minutes = int(r_time/60 - hours*60)
            self.start = time.time()
            if self.iterations != 100:
                print('iter: {0} | Loss: {1:.3f} | Remaining Time: {2} hours {3} minutes'.format(self.iterations, self.tf_loss, hours, minutes))
        return None
