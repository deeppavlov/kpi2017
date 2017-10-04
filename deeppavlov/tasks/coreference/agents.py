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

import os
from os.path import join
import copy
import random
from parlai.core.agents import Teacher
from .build import build
from . import utils

class BaseTeacher(Teacher):
    
    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('Coreference Teacher')
        group.add_argument('--random-seed', default=None)
        group.add_argument('--split', type=float, default=0.2)
        group.add_argument('--language', type=str, default='russian')
    
    def __init__(self, opt, shared=None):
        
        self.language = opt['language']
        self.id = 'coreference_teacher'

        # store datatype
        self.dt = opt['datatype'].split(':')[0]
        self.datapath = join(opt['datapath'], 'coreference', self.language)
        self.reports_datapath = join(self.datapath, 'report')
        self.scorer_path = join(self.datapath, 'scorer/reference-coreference-scorers/v8.01/scorer.pl')       
        
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
        self.epochDone = False
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
        datafile = join(self.datapath, self.doc_address[self.doc_id])
        if self.doc_id == self.len - 1:
            act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, self.doc_address[self.doc_id], epoch_done=True)
        else:
            act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, self.doc_address[self.doc_id])
   
        return act_dict
            
    def observe(self, observation):
        self.observation = copy.deepcopy(observation)
        if self.observation['epoch_done']:
            self.doc_id = 0
            self.epoch += 1
            random.shuffle(self.doc_address)
            self.epochDone = True
        else:
            self.doc_id = int(self.observation['iter_id']) + 1
            self.iter += 1
            
        if self.observation['conll']:
            predict = os.path.join(self.reports_datapath, 'response_files', self.doc_address[int(self.observation['iter_id'])])
            utils.dict2conll(self.observation, predict)  # predict it is file name
        return None

    def report(self):  # not done yet
        print('End epoch ...')
        scorer = self.scorer_path
        predicts_path = os.path.join(self.reports_datapath, 'response_files')
        keys_path = self.datapath
        results = utils.score(scorer, keys_path, predicts_path)
        
        resp_list = os.listdir(predicts_path)
        resu_list = os.listdir(os.path.join(self.reports_datapath, 'results'))
        for x in resp_list:
            os.remove(os.path.join(predicts_path, x))
        for x in resu_list:
            os.remove(os.path.join(self.reports_datapath, 'results', x))

        return results
    
    def reset(self):
        self.doc_id = 0 
        self.iter = 0        
        self.episode_done = True
        self.epochDone = False
