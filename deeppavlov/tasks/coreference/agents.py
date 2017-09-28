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
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, epoch_done=True)
            else:
                datafile = os.path.join(self.train_datapath, self.train_doc_address[self.train_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt)
            return act_dict
        elif self.dt == 'test':
            if self.test_doc_id == self.test_len - 1:
                datafile = os.path.join(self.test_datapath, self.test_doc_address[self.test_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, epoch_done=True)
            else:
                datafile = os.path.join(self.test_datapath, self.test_doc_address[self.test_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt)
        elif self.dt == 'valid':
            if self.valid_doc_id == self.valid_len - 1:
                datafile = os.path.join(self.valid_datapath, self.valid_doc_address[self.valid_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt, epoch_done=True)
            else:
                datafile = os.path.join(self.valid_datapath, self.valid_doc_address[self.valid_doc_id])
                act_dict = utils.conll2dict(self.iter, datafile, self.id, self.dt)
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

        output_file_path = os.path.join(self.reports_datapath, 'response_files')
        report = os.path.join(self.reports_datapath, 'results')

        for file in os.listdir(output_file_path):
            if self.dt == 'train':
                key_file_path = os.path.join(self.train_datapath, file)
            elif self.dt == 'valid':
                key_file_path = os.path.join(self.valid_datapath, file)
            elif self.dt == 'test':
                key_file_path = os.path.join(self.test_datapath, file)
            else:
                raise ValueError('Unknown mode: {}. Available modes: train, test.'.format(self.dt))
            report_file_path = '{0}.{1}'.format(os.path.join(report, basename(key_file_path)), 'scor')
            cmd = '{0} {1} {2} {3} none > {4}'.format(self.scorer_path, 'all', key_file_path, output_file_path, report_file_path)
            os.system(cmd)

        print('score: aggregating results...')
        k = 0
        results = dict()

        f1 = []
        for metric in ['muc', 'bcub', 'ceafm', 'ceafe']:
            recall = []
            precision = []
            for key_file in os.listdir(report):
                with open(os.path.join(report,key_file), 'r') as score_file:
                    lines = score_file.readlines()
                    if lines[-1].strip() != '--------------------------------------------------------------------------':
                        continue

                    coreference_scores_line = lines[-2]
                    tokens = coreference_scores_line.replace('\t', ' ').split()
                    r1 = float(tokens[2].strip('()'))
                    r2 = float(tokens[4].strip('()'))
                    p1 = float(tokens[7].strip('()'))
                    p2 = float(tokens[9].strip('()'))
                    if r2 == 0 or p2 == 0:
                        continue
                    recall.append((r1, r2))
                    precision.append((p1, p2))
                    k += 1

            r1 = sum(map(lambda x: x[0], recall))
            r2 = sum(map(lambda x: x[1], recall))
            p1 = sum(map(lambda x: x[0], precision))
            p2 = sum(map(lambda x: x[1], precision))
            r, p = r1 / r2, p1 / p2
            f = (2 * p * r) / (p + r)
            f1.append(f)
            print(
                '{0} precision: ({1:.3f}/{2}) {3:.3f}\t recall: ({4:.3f}/{5}) {6:.3f}\t F-1: {7:.5f}'.format(metric, p1,
                                                                                                             p2, p, r1,
                                                                                                             r2, r, f))
            results[metric] = {'p': p, 'r': r, 'f-1': f}

        key_files = os.listdir(report)
        print('avg: {0:.5f}'.format(np.mean(f1)))
        # muc bcub ceafe
        conllf1 = np.mean(f1[:2] + f1[-1:])
        print('conll F-1: {0:.5f}'.format(conllf1))
        print('using {}/{}'.format(k, 4 * len(key_files)))
        results['avg F-1'] = np.mean(f1)
        results['conll F-1'] = conllf1

        os.removedirs(output_file_path)
        os.removedirs(report)

        return results
