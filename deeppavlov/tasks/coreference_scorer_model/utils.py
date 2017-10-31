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

import parlai.core.build_data as build_data
from sklearn.model_selection import train_test_split


def get_doc_name(line):
    '''
    returns doc_name from #begin line
    '''
    return line.split('(')[1].split(')')[0]


def train_valid_test_split(inpath, train_path, valid_path, test_path, valid_ratio, test_ratio, seed=None):
    '''
    split dataset on train/valid/test
    '''
    assert valid_ratio + test_ratio <= 1.0
    assert valid_ratio > 0 and test_ratio > 0
    source_files = list(sorted(os.listdir(inpath)))

    train_valid, test = train_test_split(source_files, test_size=test_ratio, random_state=seed)
    train, valid = train_test_split(train_valid, test_size=valid_ratio / (1 - test_ratio), random_state=seed)

    print('train_valid_test_split: {}/{}/{}'.format(len(train), len(valid), len(test)))
    for dataset, data_path in zip([train, valid, test], [train_path, valid_path, test_path]):
        for el in dataset:
            build_data.move(os.path.join(inpath, el), os.path.join(data_path, el))
    return None


def save_observations(observation, path_to_save, lang='ru'):
    for lines in observation:
        doc_name = get_doc_name(lines[0])
        with open(os.path.join(path_to_save, doc_name + '.{}.v4_conll'.format(lang)), 'w') as fout:
            for line in lines:
                fout.write(line)
