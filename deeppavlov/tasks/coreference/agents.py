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


import os
from os.path import join
import copy
import random
from parlai.core.agents import Teacher
from shutil import copy as cp

from .build import build
from . import utils
import tensorflow as tf
from ...utils import coreference_utils


class CoreferenceTeacher(Teacher):
    """Teacher for coreference resolution task"""
    
    @staticmethod
    def add_cmdline_args(argparser):
        """Parameters of agent and default values"""
        group = argparser.add_argument_group('Coreference Teacher')
        group.add_argument('--random-seed', type=int, default=0)
        group.add_argument('--split', type=float, default=0.2)
        group.add_argument('--language', type=str, default='russian')
        group.add_argument('--log_root', type=str, default='./build/coreference/')

    def __init__(self, opt, shared=None):
        """Initialize the parameters for CoreferenceTeacher"""
        
        self.language = opt['language']
        self.id = 'coreference_teacher'
        build(opt)
        # store datatype
        self.dt = opt['datatype'].split(':')[0]
        self.datapath = join(opt['datapath'], 'coreference', self.language)
        self.reports_datapath = join(opt['log_root'], self.language, 'agent', 'reports')
        self.scorer_path = join(self.datapath, 'scorer/reference-coreference-scorers/v8.01/scorer.pl')
        random.seed(opt['random_seed'])
        
        if self.dt == 'train':
            self.datapath = join(self.datapath, 'train')
        elif self.dt == 'valid':
            self.datapath = join(self.datapath, 'valid')
        elif self.dt == 'test':
            self.datapath = join(self.datapath, 'test')
        else:
            raise ValueError('Unknown mode: {}. Available modes: train, test, valid.'.format(self.dt))
        
        self.doc_address = os.listdir(self.datapath)  # list of files addresses        
        self.len = len(self.doc_address)
        self.doc_id = 0 
        self.iter = 0
        self.epoch = 0
        self.episode_done = False
        self.epochDone = False
        self.writer = tf.summary.FileWriter(join(opt['log_root'], self.language, 'agent', 'logs', opt['name']))
        super().__init__(opt, shared)
    
    def __len__(self):
        return self.len

    def __iter__(self):
        self.epochDone = False
        return self

    def __next__(self):
        if self.epochDone:
            raise StopIteration()    

    def act(self):
        """reads document and returns it"""
        datafile = join(self.datapath, self.doc_address[self.doc_id])
        epoch_done = self.doc_id == self.len - 1
        act_dict = utils.conll2dict(datafile, self.iter, self.id, self.dt, self.doc_address[self.doc_id],
                                    epoch_done=epoch_done)
   
        return act_dict
            
    def observe(self, observation):
        """saves observation"""
        self.observation = copy.deepcopy(observation)
        if self.observation['epoch_done']:
            self.doc_id = 0
            self.epoch += 1
            random.shuffle(self.doc_address)
            self.epochDone = True
        else:
            self.doc_id = int(self.observation['iter_id']) + 1
            self.iter += 1
            if self.dt == 'train':
                summary_dict = {'Loss': self.observation['loss']}
                step = self.observation['tf_step']
                utils.summary(summary_dict, step, self.writer)
            
        if self.observation['conll']:
            predict_path = os.path.join(self.reports_datapath, 'response_files',
                                        self.doc_address[int(self.observation['iter_id'])])
            utils.dict2conll(self.observation, predict_path)  # predict it is file name
        return self.observation

    def report(self):
        """calls coreference and anaphora scorer on last observation and reports result"""
        scorer = self.scorer_path
        predicts_path = os.path.join(self.reports_datapath, 'response_files')
        keys_path = self.datapath

        r = coreference_utils.score(scorer, keys_path, predicts_path)
        z = utils.anaphora_score(keys_path, predicts_path)
        r['anaphora_precision'] = z['precision']
        r['anaphora_recall'] = z['recall']
        r['anaphora_F1'] = z['F1']

        step = self.observation['iteration']
        summary_dict = {'f1': r['conll-F-1'], 'avg-F-1': r['avg-F-1'], 'anaphora_F1': r['anaphora_F1']}

        utils.summary(summary_dict, step, self.writer)

        resp_list = os.listdir(predicts_path)
        resu_list = os.listdir(os.path.join(self.reports_datapath, 'results'))
        pred_old_list = os.listdir(os.path.join(self.reports_datapath, 'predictions'))

        for x in pred_old_list:
            os.remove(os.path.join(self.reports_datapath, 'predictions', x))

        for x in resu_list:
            os.remove(os.path.join(self.reports_datapath, 'results', x))

        for x in resp_list:
            if x.endswith('conll'):
                cp(os.path.join(predicts_path, x), os.path.join(self.reports_datapath, 'predictions'))
            else:
                os.remove(os.path.join(predicts_path, x))

        return r
    
    def reset(self):
        self.doc_id = 0 
        self.iter = 0        
        self.episode_done = True
        self.epochDone = False


class DefaultTeacher(CoreferenceTeacher):
    pass
