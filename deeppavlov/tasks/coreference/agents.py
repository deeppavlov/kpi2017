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
from os.path import basename
import numpy as np
import copy
from parlai.core.agents import Teacher
from .build import build
from . import utils

class BaseTeacher(Teacher):
    
    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('Coreference Teacher')
        group.add_argument('--data-path', default=None, help='path to rucorp/Ontonotes dataset')
        group.add_argument('--random-seed', default=None)
        group.add_argument('--split', type=float, default=0.2)
        group.add_argument('--cor', type=str, default='coreference')
        group.add_argument('--language', type=str, default='russian')
    
    def __init__(self, opt, shared=None):
        
        self.task = opt['cor']  # 'coreference'
        self.language = opt['language']
        self.id = 'coreference_teacher'

        # store datatype
        self.dt = opt['datatype'].split(':')[0]
        build(opt)
        
        self.scorer_path = os.path.join(opt['datapath'], self.task, self.language, 'scorer/reference-coreference-scorers/v8.01/scorer.pl')
        self.train_datapath = os.path.join(opt['datapath'], self.task, self.language, 'train')
        self.test_datapath = os.path.join(opt['datapath'], self.task, self.language, 'test')
        self.valid_datapath = os.path.join(opt['datapath'], self.task, self.language, 'valid')
        self.train_doc_address = os.listdir(self.train_datapath)  # list of files addresses
        self.test_doc_address = os.listdir(self.test_datapath)
        self.valid_doc_address = os.listdir(self.valid_datapath)
        self.train_len = len(self.train_doc_address)
        self.test_len = len(self.test_doc_address)
        self.valid_len = len(self.valid_doc_address)
        self.train_doc_id = 0
        self.test_doc_id = 0
        self.valid_doc_id = 0
        self.iter = 0
        self.epoch = 0
        self.epochDone = False
        self.reports_datapath = os.path.join(opt['datapath'], self.task, self.language, 'report')
        super().__init__(opt, shared)
    
    def __len__(self):
        if self.dt == 'train':
            return self.train_len
        if self.dt == 'test':
            return self.test_len
        if self.dt == 'valid':
            return self.valid_len

    def __iter__(self):
        self.epochDone = False
        return self

    def __next__(self):
        if self.epochDone:
            raise StopIteration()    
    
    
    def act(self):
        if self.dt == 'train':
            if self.train_doc_id == self.train_len - 1:
                datafile = os.path.join(self.train_datapath, self.train_doc_address[self.train_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, self.train_doc_address[self.train_doc_id], epoch_done=True)
            else:
                datafile = os.path.join(self.train_datapath, self.train_doc_address[self.train_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, self.train_doc_address[self.train_doc_id])
            return act_dict
        elif self.dt == 'test':
            if self.test_doc_id == self.test_len - 1:
                datafile = os.path.join(self.test_datapath, self.test_doc_address[self.test_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, self.train_doc_address[self.test_doc_id], epoch_done=True)
            else:
                datafile = os.path.join(self.test_datapath, self.test_doc_address[self.test_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, self.train_doc_address[self.test_doc_id])
        elif self.dt == 'valid':
            if self.valid_doc_id == self.valid_len - 1:
                datafile = os.path.join(self.valid_datapath, self.valid_doc_address[self.valid_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, self.train_doc_address[self.valid_doc_id], epoch_done=True)
            else:
                datafile = os.path.join(self.valid_datapath, self.valid_doc_address[self.valid_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, self.train_doc_address[self.valid_doc_id])
        else:
            raise ValueError('Unknown mode: {}. Available modes: train, test, valid.'.format(self.dt))
        return act_dict
            
    def observe(self, observation):
        self.observation = copy.deepcopy(observation)
        if self.dt == 'train':
            if self.observation['epoch_done']:
                self.train_doc_id = 0
                self.epoch += 1
                self.epochDone = True
            else:
                self.train_doc_id = int(self.observation['iter_id']) + 1
                self.iter += 1
            if self.observation['conll']:
                predict = os.path.join(self.reports_datapath, 'response_files', self.train_doc_address[int(self.observation['iter_id'])])
                utils.dict2conll(self.observation, predict)  # predict it is file name

        elif self.dt == 'test':
            if self.observation['epoch_done']:
                self.test_doc_id = 0
                self.epochDone = True
            else:
                self.test_doc_id = int(self.observation['iter_id']) + 1
            predict = os.path.join(self.reports_datapath, 'response_files', self.test_doc_address[int(self.observation['iter_id'])])
            utils.dict2conll(self.observation, predict)  # predict it is file name

        elif self.dt == 'valid':
            if self.observation['epoch_done']:
                self.valid_doc_id = 0
            else:
                self.valid_doc_id = int(self.observation['iter_id']) + 1
            predict = os.path.join(self.reports_datapath, 'response_files', self.valid_doc_address[int(self.observation['iter_id'])])
            utils.dict2conll(self.observation, predict)  # predict it is file name
        else:
            raise ValueError('Unknown mode: {}. Available modes: train, test.'.format(self.dt))
        return None

    def report(self):  # not done yet
        print('End epoch ...')
        scorer = self.scorer_path
        predicts_path = os.path.join(self.reports_datapath, 'response_files')
        if self.dt == 'train':
            keys_path = self.train_datapath
            results = utils.score(scorer, keys_path, predicts_path)
        elif self.dt == 'valid':
            keys_path = self.valid_datapath
            results = utils.score(scorer, keys_path, predicts_path)
        elif self.dt == 'test':
            keys_path = self.test_datapath
            results = utils.score(scorer, keys_path, predicts_path)
        else:
            raise ValueError('Unknown mode: {}. Available modes: train, test.'.format(self.dt))
        
        resp_list = os.listdir(predicts_path)
        resu_list = os.listdir(os.path.join(self.reports_datapath, 'results'))
        for x in resp_list:
            os.remove(os.path.join(predicts_path, x))
        for x in resu_list:
            os.remove(os.path.join(self.reports_datapath, 'results', x))

        return results
