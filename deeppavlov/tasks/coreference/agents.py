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
import copy
import re
import subprocess
from parlai.core.dialog_teacher import Teacher
from .build import build

COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)

def conll2dict(t_id,conll, epoch_done=False):
    data = {'doc_id': [],
            'part_id': [],
            'word_number': [],
            'word': [],
            'part_of_speech': [],
            'parse_bit': [],
            'lemma': [],
            'sense': [],
            'speaker': [],
            'entiti': [],
            'predict': [],
            'coreference': [],
            'id': t_id,
            'epoch_done': False}

    with open(conll,'r') as f:
        for line in f:
            row = line.split()
            if row[0].startswith('#'):
                pass
            assert len(row) >= 12
            data['doc_id'].append(row[0])
            data['part_id'].append(row[1])
            data['word_number'].append(row[2])
            data['word'].append(row[3])
            data['part_of_speech'].append(row[4])
            data['parse_bit'].append(row[5])
            data['lemma'].append(row[6])
            data['sense'].append(row[7])
            data['speaker'].append(row[8])
            data['entiti'].append(row[9])
            data['predict'].append(row[10])
            data['coreference'].append(row[11])
        conll.close()
    return data

def dict2conll(data, predict):
    #
    with open(predict,'w') as CoNLL:
        for i in range(len(data['doc_id'])):
            if i == 0:
                CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i], data["part_id"][i]))
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                    data["part_id"][i],
                                                    data["word_number"][i],
                                                    data["word"][i],
                                                    data["part_of_speech"][i],
                                                    data["parse_bit"][i],
                                                    data["lemma"][i],
                                                    data["sense"][i],
                                                    data["speaker"][i],
                                                    data["entiti"][i],
                                                    data["predict"][i],
                                                    data["coref"][i]))
            elif i == len(data['doc_id'])-1:
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                    data["part_id"][i],
                                                    data["word_number"][i],
                                                    data["word"][i],
                                                    data["part_of_speech"][i],
                                                    data["parse_bit"][i],
                                                    data["lemma"][i],
                                                    data["sense"][i],
                                                    data["speaker"][i],
                                                    data["entiti"][i],
                                                    data["predict"][i],
                                                    data["coref"][i]))
                CoNLL.write('\n')
                CoNLL.write('#end document\n')
            else:
                if data['doc_id'][i] == data['doc_id'][i+1]:
                    CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                        data["part_id"][i],
                                                        data["word_number"][i],
                                                        data["word"][i],
                                                        data["part_of_speech"][i],
                                                        data["parse_bit"][i],
                                                        data["lemma"][i],
                                                        data["sense"][i],
                                                        data["speaker"][i],
                                                        data["entiti"][i],
                                                        data["predict"][i],
                                                        data["coref"][i]))
                    if data["word_number"][i+1] == 0:
                        CoNLL.write('\n')
                else:
                    CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                        data["part_id"][i],
                                                        data["word_number"][i],
                                                        data["word"][i],
                                                        data["part_of_speech"][i],
                                                        data["parse_bit"][i],
                                                        data["lemma"][i],
                                                        data["sense"][i],
                                                        data["speaker"][i],
                                                        data["entiti"][i],
                                                        data["predict"][i],
                                                        data["coref"][i]))
                    CoNLL.write('\n')
                    CoNLL.write('#end document\n')
                    CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i+1], data["part_id"][i+1]))
        CoNLL.close()
    return None

def official_conll_eval(scorer_path, gold_path, predicted_path, metric, official_stdout=False):
    cmd = [scorer_path, metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    if stderr is not None:
        print(stderr)

    if official_stdout:
        print("Official result for {}".format(metric))
        print(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return { "r": recall, "p": precision, "f": f1 }

def evaluate_conll(scorer_path, gold_path, predicted_path, official_stdout=False):
    return { m: official_conll_eval(scorer_path,gold_path,predicted_path,m,official_stdout) for m in ("muc", "bcub", "ceafe")}

class BaseTeacher(Teacher):

    def __init__(self, opt, shared=None):
        
        self.task = opt['task'] # 'coreference'
        self.languange = opt['languange']
        self.id = 'coreference_teacher'
        self.mode = 'train' #or test
#         self.random_seed = opt['random_seed'] #None
#         self.split = opt['split'] # 0.2

        # store datatype
        build(opt)
        
        
        opt['datafile_train'] = os.path.join(opt['datapath'], self.task, self.language, 'train')
        opt['datafile_test'] = os.path.join(opt['datapath'], self.task, self.language, 'test')
        self.train_doc_address = os.listdir(opt['datafile_train']) # list of files addresses
        self.test_doc_address = os.listdir(opt['datafile_test']) # list of files addresses
        self.train_len = len(self.train_doc_adress)
        self.test_len = len(self.test_doc_adress)
        self.train_doc_id = 0
        self.test_doc_id = 0
        self.iter = 0
        self.epoch = 0
#         self.loss = 0
#         self.F1 = 0
        super().__init__(opt, shared)
        
    def act(self):
        if self.mode == 'train':
            if self.train_doc_id == self.train_len - 1:
                datafile = train_doc_address[self.train_doc_id]
                act_dict = conll2dict(datafile, epoch_done=True)
            else:
                datafile = train_doc_address[self.train_doc_id]
                act_dict = conll2dict(datafile)
            return act_dict
        elif self.mode == 'test':
            if self.test_doc_id == self.test_len - 1:
                datafile = test_doc_address[self.test_doc_id]
                act_dict = conll2dict(datafile, epoch_done=True)
            else:
                datafile = test_doc_address[self.test_doc_id]
                act_dict = conll2dict(datafile)
        else:
            raise TypeError('Unknown mode: {}. Available modes: train, test.'.format(self.mode))
        return act_dict
            
    def observe(self, observation):
        self.observation = copy.deepcopy(observation)
        if self.mode == 'train':
            if self.epoch_done == True:
                self.train_doc_id = 0
                self.epoch += 1
            else:
                self.train_doc_id = int(self.observation['id']) + 1
                self.iter += 1
                
            predict = os.path.join(opt['datapath'], self.task, self.language, 'report', train_doc_address[self.train_doc_id])
            dict2conll(self.observation, predict) # predict it is file name
        elif self.mode == 'test':
            if self.epoch_done == True:
                self.test_doc_id = 0
#                 self.report()
            else:
                self.test_doc_id = int(self.observation['id']) + 1
            predict = os.path.join(opt['datapath'], self.task, self.language, 'report', test_doc_address[self.test_doc_id])
            dict2conll(self.observation, predict) # predict it is file name
        else:
            raise TypeError('Unknown mode: {}. Available modes: train, test.'.format(self.mode))
        return None    

    def report(self, opt): # not done yet
        metrics = evaluate_conll(scorer_path, gold_path, predicted_path, official_stdout=False)
        return metrics
