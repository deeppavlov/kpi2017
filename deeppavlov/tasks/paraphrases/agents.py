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


from parlai.core.dialog_teacher import DialogTeacher

from .metric import BinaryClassificationMetrics
from .build import build
import os
import csv
from sklearn.model_selection import KFold
import random


def _path(opt):
    # ensure data is built
    build(opt)

    # set up paths to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    fname = 'paraphrases'
    if dt == 'test':
        fname += '_gold'
    fname += '.tsv'
    datafile = os.path.join(opt['datapath'], 'paraphrases', fname)
    return datafile


class DefaultTeacher(DialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        teacher = argparser.add_argument_group('paraphrases teacher arguments')
        teacher.add_argument('--teacher-random-seed', type=int, default=71)
        teacher.add_argument('--bagging-fold-index', type=int)
        teacher.add_argument('--bagging-folds-number', type=int, default=5)

    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype_strict = opt['datatype'].split(':')[0]

        opt['datafile'] = _path(opt)

        # store identifier for the teacher in the dialog
        self.id = 'paraphrases_teacher'

        # define standard question, since it doesn't change for this task
        self.question = "Эти два предложения — парафразы?"
        self.answer_candidates = ['Да', 'Нет']

        random_state = random.getstate()
        random.seed(opt.get('teacher_random_seed'))
        self.random_state = random.getstate()
        random.setstate(random_state)

        if shared and shared.get('metrics'):
            self.metrics = shared['metrics']
        else:
            self.metrics = BinaryClassificationMetrics('Да')

        super().__init__(opt, shared)

    def label_candidates(self):
        return self.answer_candidates

    def setup_data(self, path):
        print('loading: ' + path)

        questions = []
        y = []

        # open data file with labels
        # (path will be provided to setup_data from opt['datafile'] defined above)
        with open(path) as labels_file:
            tsv_reader = csv.reader(labels_file, delimiter='\t')

            for row in tsv_reader:
                if len(row) != 3:
                    print('Warn: expected 3 columns in a tsv row, got ' + str(row))
                    continue
                y.append(['Да' if row[0] == '1' else 'Нет'])
                questions.append(row[1] + '\n' + row[2])

        episode_done = True
        if not y:
            y = [None for _ in range(len(questions))]

        indexes = range(len(questions))
        if self.datatype_strict != 'test':
            random_state = random.getstate()
            random.setstate(self.random_state)
            kf_seed = random.randrange(500000)
            kf = KFold(self.opt.get('bagging_folds_number'), shuffle=True,
                       random_state=kf_seed)
            i = 0
            for train_index, test_index in kf.split(questions):
                indexes = train_index if self.datatype_strict == 'train' else test_index
                if i >= self.opt.get('bagging_fold_index', 0):
                    break
            self.random_state = random.getstate()
            random.setstate(random_state)

        # define iterator over all queries
        for i in indexes:
            # get current label, both as a digit and as a text
            # yield tuple with information and episode_done? flag
            yield (self.question + "\n" + questions[i], y[i]), episode_done

    def reset(self):
        super().reset()

        random_state = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(self.data.data)
        self.random_state = random.getstate()
        random.setstate(random_state)
