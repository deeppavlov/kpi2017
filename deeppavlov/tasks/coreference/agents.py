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
import tensorflow as tf

class DefaultTeacher(Teacher):
    
    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('Coreference Teacher')
        group.add_argument('--random-seed', default=None)
        group.add_argument('--split', type=float, default=0.2)
        group.add_argument('--language', type=str, default='russian')
        group.add_argument('--log_root', type=str, default='./build/coreference/')
    
    def __init__(self, opt, shared=None):
        
        self.language = opt['language']
        self.id = 'coreference_teacher'
        build(opt)
        # store datatype
        self.dt = opt['datatype'].split(':')[0]
        self.datapath = join(opt['datapath'], 'coreference', self.language)
        self.reports_datapath = join(opt['log_root'], self.language, 'agent','reports')
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
        datafile = join(self.datapath, self.doc_address[self.doc_id])
        epoch_done = self.doc_id == self.len - 1
        act_dict = utils.conll2dict(datafile, self.iter, self.id, self.dt, self.doc_address[self.doc_id],
                                    epoch_done=epoch_done)
   
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
            if self.dt == 'train':
                summary_dict = {'Loss': self.observation['loss']}
                step = self.observation['iteration']
                utils.summary(summary_dict, step, self.writer)
            
        if self.observation['conll']:
            predict_path = os.path.join(self.reports_datapath, 'response_files',
                                   self.doc_address[int(self.observation['iter_id'])])
            utils.dict2conll(self.observation, predict_path)  # predict it is file name
        return self.observation

    def report(self):
        scorer = self.scorer_path
        predicts_path = os.path.join(self.reports_datapath, 'response_files')
        keys_path = self.datapath
        r = utils.score(scorer, keys_path, predicts_path)
        step = self.observation['iteration']
        summary_dict = {'f1': r['f1'], 'avg-F-1': r['avg-F-1']}
        utils.summary(summary_dict, step, self.writer)
        
        resp_list = os.listdir(predicts_path)
        resu_list = os.listdir(os.path.join(self.reports_datapath, 'results'))
        for x in resp_list:
            os.remove(os.path.join(predicts_path, x))
        for x in resu_list:
            os.remove(os.path.join(self.reports_datapath, 'results', x))
        
        s = '\n {0}\n {1}\n {2}\n {3}\n {4}\n avg-F-1: {5}\n f1: {6}\n'.format(r['muc'],r['bcub'],r['ceafm'],r['ceafe'],
                                                           r['using'],r['avg-F-1'],r['f1'])
        return r
    
    def reset(self):
        self.doc_id = 0 
        self.iter = 0        
        self.episode_done = True
        self.epochDone = False
