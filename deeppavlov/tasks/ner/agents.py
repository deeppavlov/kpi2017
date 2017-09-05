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

from parlai.core.dialog_teacher import DialogTeacher
from .build import build
import os
import xml.etree.ElementTree as ET
import random
from .metric import BinaryClassificationMetrics


def _path(opt):
    # ensure data is built
    build(opt)
    fname = 'heap.txt'
    datafile = os.path.join(opt['datapath'], fname)
    return datafile


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        assert opt['train_part'] + opt['test_part'] + opt['valid_part'] == 1
        self.parts = [opt['train_part'], opt['valid_part'], opt['test_part']]
        random.seed(opt['teacher_seed'])
        # store datatype
        self.dt = opt['datatype'].split(':')[0]
        self.opt = opt
        opt['datafile'] = _path(opt)

        # store identifier for the teacher in the dialog
        self.id = 'ner_teacher'

        if shared and shared.get('metrics'):
            self.metrics = shared['metrics']
        else:
            self.metrics = BinaryClassificationMetrics(opt['model_file'])

        # define standard question, since it doesn't change for this task
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('NER Teacher')
        group.add_argument('--raw-data-path', default=None,
                           help='path to Gareev dataset')
        group.add_argument('--teacher-seed', type=int, default=42)
        group.add_argument('--train-part', type=int, default=0.8)
        group.add_argument('--valid-part', type=int, default=0.1)
        group.add_argument('--test-part', type=int, default=0.1)

    def setup_data(self, path):
        print('loading: ' + path)

        questions = []
        y = []

        # open data file with labels
        # (path will be provided to setup_data from opt['datafile'] defined above)

        with open(path) as heap_file:
            tokens = []
            tags = []
            for line in heap_file:
                if len(line) > 2:
                    token, tag = line.split()
                    tokens.append(token)
                    tags.append(tag)
                else:
                    questions.append(' '.join(tokens))
                    y.append([' '.join(tags)])
                    tokens = []
                    tags = []

        questions_and_ys = list(zip(questions, y))
        random.shuffle(questions_and_ys)
        questions, y = list(zip(*questions_and_ys))

        if self.dt == 'train':
            part = [0, self.parts[0]]
        elif self.dt == 'test':
            part = [self.parts[0], sum(self.parts[0:2])]
        elif self.dt == 'valid':
            part = [sum(self.parts[0:2]), 1]
        episode_done = True

        n_docs = len(questions)
        start_ind = int(n_docs * part[0])
        end_ind = int(n_docs * part[1])
        questions = questions[start_ind: end_ind]
        y = y[start_ind: end_ind]

        if not y:
            y = [None for _ in range(len(questions))]

        # define iterator over all queries
        for i in range(len(questions)):
            # get current label, both as a digit and as a text
            # yield tuple with information and episode_done? flag
            yield (questions[i], y[i]), episode_done

    def reset(self):
        random.shuffle(self.data.data)
        super().reset()
