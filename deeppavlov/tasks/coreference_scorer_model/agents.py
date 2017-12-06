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
import random

import numpy as np
import tensorflow as tf
from parlai.core.agents import Teacher

from . import utils
from .build import build
from ...utils import coreference_utils


class CoreferenceTeacher(Teacher):
    """Teacher for coreference resolution task"""

    @staticmethod
    def add_cmdline_args(argparser):
        """Parameters of agent and default values"""
        group = argparser.add_argument_group('Coreference Teacher')
        group.add_argument('--language', type=str, default='ru')
        group.add_argument('--predictions_folder', type=str, default='predicts',
                           help='folder where to dump conll predictions, scorer will use this folder')
        group.add_argument('--scorer_path', type=str, default='scorer/reference-coreference-scorers/v8.01/scorer.pl',
                           help='path to CoNLL scorer perl script')
        group.add_argument('--valid_ratio', type=float,
                           default=0.2, help='valid_set ratio')
        group.add_argument('--test_ratio', type=float,
                           default=0.2, help='test_set ratio')
        group.add_argument('--teacher_seed', type=int, default=42, help='seed')
        group.add_argument('--raw-dataset-path', type=str, default=None,
                             help='Path to folder with two subfolders: dataset and scorer. '
                                  'These two folders are extracted rucoref_29.10.2015.zip and '
                                  'reference-coreference-scorers.v8.01.tar.gz')

    def __init__(self, opt, shared=None):
        """Initialize the parameters for CoreferenceTeacher"""
        super().__init__(opt, shared)
        self.last_observation = None
        self.id = 'two-step-coref'

        self.seed = opt['teacher_seed']
        np.random.seed(seed=self.seed)
        random.seed(a=self.seed)
        tf.set_random_seed(seed=self.seed)

        if shared:
            raise RuntimeError('Additional batching is not supported')

        build(opt)

        self.dt = opt['datatype'].split(':')[0]
        self.datapath = os.path.join(opt['datapath'], 'coreference_scorer_model', opt['language'])
        self.valid_path = None
        self.train_path = None
        self.predictions_folder = os.path.join(self.datapath, opt['predictions_folder'], self.dt)
        self.scorer_path = os.path.join(self.datapath, opt['scorer_path'])

        # in train mode we use train dataset to train model
        # and valid dataset to adjust threshold
        # in valid and test mode we use test dataset
        if self.dt == 'train':
            self.valid_path = os.path.join(self.datapath, 'valid')
            self.train_path = os.path.join(self.datapath, 'train')
        elif self.dt in ['test', 'valid']:
            self.valid_path = os.path.join(self.datapath, 'test')
        else:
            raise ValueError('Unknown mode: {}. Available modes: train, test, valid.'.format(self.dt))

        self.train_documents = [] if self.train_path is None else list(sorted(os.listdir(self.train_path)))
        self.valid_documents = [] if self.valid_path is None else list(sorted(os.listdir(self.valid_path)))
        self.len = 1
        self.epoch = 0
        self._epoch_done = False

    def act(self):
        """reads all documents and returns them"""
        self._epoch_done = True
        train_conll = [open(os.path.join(self.train_path, file), 'r').readlines() for file in self.train_documents]
        valid_conll = [open(os.path.join(self.valid_path, file), 'r').readlines() for file in self.valid_documents]
        return {'id': self.id, 'conll': train_conll, 'valid_conll': valid_conll}

    def observe(self, observation):
        """saves observation"""
        self.last_observation = observation
        self.epoch += 1

    def report(self):
        """calls scorer on last observation and reports result"""
        utils.save_observations(self.last_observation['valid_conll'], self.predictions_folder)
        res = coreference_utils.score(self.scorer_path, self.valid_path, self.predictions_folder)
        return {'f1': res['conll-F-1']}

    def reset(self):
        self._epoch_done = False

    def epoch_done(self):
        return self._epoch_done

    def __len__(self):
        return self.len
