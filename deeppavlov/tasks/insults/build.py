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

import parlai.core.build_data as build_data
import os
import re
import string
import numpy as np
import pandas as pd
import urllib


def data_preprocessing(f):
    """Preprocess the data.

    Keyword arguments:
        f -- list of text samples
    """
    f = [x.lower() for x in f]
    f = [x[1:-1] for x in f]
    f = [x.replace("\\n", " ") for x in f]
    f = [x.replace("\\t", " ") for x in f]
    f = [x.replace("\\xa0", " ") for x in f]
    f = [x.replace("\\xc2", " ") for x in f]

    f = [re.sub('!!+', ' !! ', x) for x in f]
    f = [re.sub('!', ' ! ', x) for x in f]
    f = [re.sub('! !', '!!', x) for x in f]

    f = [re.sub('\?\?+', ' ?? ', x) for x in f]
    f = [re.sub('\?', ' ? ', x) for x in f]
    f = [re.sub('\? \?', '??', x) for x in f]

    f = [re.sub('\?!+', ' ?! ', x) for x in f]

    f = [re.sub('\.\.+', '..', x) for x in f]
    f = [re.sub('\.', ' . ', x) for x in f]
    f = [re.sub('\.  \.', '..', x) for x in f]

    f = [re.sub(',', ' , ', x) for x in f]
    f = [re.sub(':', ' : ', x) for x in f]
    f = [re.sub(';', ' ; ', x) for x in f]
    f = [re.sub('\%', ' % ', x) for x in f]

    f = [x.replace("$", "s") for x in f]
    f = [x.replace(" u ", " you ") for x in f]
    f = [x.replace(" em ", " them ") for x in f]
    f = [x.replace(" da ", " the ") for x in f]
    f = [x.replace(" yo ", " you ") for x in f]
    f = [x.replace(" ur ", " your ") for x in f]
    f = [x.replace("you\'re", "you are") for x in f]
    f = [x.replace(" u r ", " you are ") for x in f]
    f = [x.replace("yo\'re", " you are ") for x in f]
    f = [x.replace("yu\'re", " you are ") for x in f]
    f = [x.replace("u\'re", " you are ") for x in f]
    f = [x.replace(" urs ", " yours ") for x in f]
    f = [x.replace("y'all", "you all") for x in f]

    f = [x.replace(" r u ", " are you ") for x in f]
    f = [x.replace(" r you", " are you") for x in f]
    f = [x.replace(" are u ", " are you ") for x in f]

    f = [x.replace(" mom ", " mother ") for x in f]
    f = [x.replace(" momm ", " mother ") for x in f]
    f = [x.replace(" mommy ", " mother ") for x in f]
    f = [x.replace(" momma ", " mother ") for x in f]
    f = [x.replace(" mama ", " mother ") for x in f]
    f = [x.replace(" mamma ", " mother ") for x in f]
    f = [x.replace(" mum ", " mother ") for x in f]
    f = [x.replace(" mummy ", " mother ") for x in f]

    f = [x.replace("won't", "will not") for x in f]
    f = [x.replace("can't", "cannot") for x in f]
    f = [x.replace("i'm", "i am") for x in f]
    f = [x.replace(" im ", " i am ") for x in f]
    f = [x.replace("ain't", "is not") for x in f]
    f = [x.replace("'ll", " will") for x in f]
    f = [x.replace("'t", " not") for x in f]
    f = [x.replace("'ve", " have") for x in f]
    f = [x.replace("'s", " is") for x in f]
    f = [x.replace("'re", " are") for x in f]
    f = [x.replace("'d", " would") for x in f]

    # replace multiple letters (3 and more) by 2 letters
    for letter in string.ascii_lowercase:
        f = [re.sub(letter * 3 + '+', letter, x).strip() for x in f]

    bad_words_file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "badwords.txt"), "r")
    bwMap = dict()
    for line in bad_words_file:
        sp = line.strip().lower().split(",")
        if len(sp) == 2:
            bwMap[sp[0].strip()] = sp[1].strip()

    for key, value in bwMap.items():
        kpad = " " + key + " "
        vpad = " " + value + " "
        f = [x.replace(kpad, vpad) for x in f]

    # stemming
    f = [re.sub("ies( |$)", "y ", x) for x in f]
    f = [re.sub("s( |$)", " ", x) for x in f]
    f = [re.sub("ing( |$)", " ", x) for x in f]
    f = [x.replace("tard ", " ") for x in f]

    f = [re.sub(" [*$%&#@][*$%&#@]+", " xexp ", x) for x in f]
    f = [re.sub(" [0-9]+ ", " DD ", x) for x in f]
    f = [re.sub("<\S*>", "", x) for x in f]
    f = [re.sub('\s+', ' ', x) for x in f]
    return f


def write_input_fasttext_cls(data, path, data_name):
    """Write down input files for fasttext classificator.

    Keyword arguments:
        data -- array of text samples
        path -- path to folder to put the files
        data_name -- mode of writing files "train" or "test"
    """
    f = open(path + '_fasttext_cls.txt', 'w')
    for i in range(data.shape[0]):
        if data_name == 'train':
            f.write('__label__' + str(data.iloc[i,0]) + ' ' + data.iloc[i,1] + '\n')
        elif data_name == 'test':
            f.write(data.iloc[i,1] + '\n')
        else:
            print('Incorrect data name')
    f.close()


def write_input_fasttext_emb(data, path, data_name):
    """Write down input files for fasttext embedding.

    Keyword arguments:
        data -- array of text samples
        path -- path to folder to put the files
        data_name -- mode of writing files "train" or "test"
    """
    f = open(path + '_fasttext_emb.txt', 'w')
    for i in range(data.shape[0]):
        if data_name == 'train' or data_name == 'test':
            f.write(data.iloc[i,1] + '\n')
        else:
            print('Incorrect data name')
    f.close()


def balance_dataset(dataset_0, labels_0, dataset_1, labels_1, ratio=1):
    """Balance the dataset_0 with samples from dataset_1 up to given ratio.

    Keyword arguments:
        dataset_0 -- array of text samples
        labels_0 -- array of labels for dataset_0
        dataset_1 -- array of text samples
        labels_1 -- array of labels for dataset_1
        ratio -- ratio of samples of class 1 to samples of class 0 (default 1.0)
    """
    initial_train_size = dataset_0.shape[0]
    insult_inds = np.nonzero(labels_1)[0]
    num_insults_0 = len(np.nonzero(labels_0)[0])
    num_insults_1 = len(np.nonzero(labels_1)[0])
    insult_inds_to_add = insult_inds[np.random.randint(low=0, high=num_insults_1,
                                                       size=(ratio * (initial_train_size - num_insults_0) - num_insults_0))]
    result = dataset_0.append(dataset_1.iloc[insult_inds_to_add])
    result_labels = labels_0.append(labels_1.iloc[insult_inds_to_add])
    return result, result_labels


def build(opt):
    """Read and preprocess data, save preprocessed data, balance data,
    create input files for fasttext classifier and embeddings.

    Keyword arguments:
        opt -- given parameters
    """
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'insults')
    # define version if any
    version = '1.0'

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        raw_path = os.path.abspath(opt['raw_dataset_path'] or ".")
        train_file = os.path.join(raw_path, 'train.csv')
        valid_file = os.path.join(raw_path, 'test_with_solutions.csv')
        test_file = os.path.join(raw_path, 'impermium_verification_labels.csv')
        if not os.path.isfile(train_file) or not os.path.isfile(valid_file) or not os.path.isfile(test_file):
            ds_path = os.environ.get('DATASETS_URL')
            file_name = 'insults.tar.gz'
            if not ds_path:
                raise RuntimeError('Please download dataset files from'
                                   ' https://www.kaggle.com/c/detecting-insults-in-social-commentary/data'
                                   ' and set path to their directory in raw-dataset-path parameter')
            print('Trying to download a insults dataset from the repository')
            url = urllib.parse.urljoin(ds_path, file_name)
            print(repr(url))
            build_data.download(url, dpath, file_name)
            build_data.untar(dpath, file_name)
            opt['raw_dataset_path'] = dpath
            print('Downloaded a insults dataset')

            raw_path = os.path.abspath(opt['raw_dataset_path'])
            train_file = os.path.join(raw_path, 'train.csv')
            valid_file = os.path.join(raw_path, 'test_with_solutions.csv')
            test_file = os.path.join(raw_path, 'impermium_verification_labels.csv')

        train_data = pd.read_csv(train_file)
        train_data = train_data.drop('Date', axis=1)

        test_data = pd.read_csv(test_file)
        test_data = test_data.drop('id', axis=1)
        test_data = test_data.drop('Usage', axis=1)
        test_data = test_data.drop('Date', axis=1)

        valid_data = pd.read_csv(valid_file)
        valid_data = valid_data.drop('Date', axis=1)
        valid_data = valid_data.drop('Usage', axis=1)

        # merge train and valid due to use of cross validation
        train_data = train_data.append(valid_data)

        if opt.get('balance_train_dataset'):
            if opt['balance_train_dataset'] == True:
                train_data['Comment'],train_data['Insult'] = balance_dataset(train_data['Comment'],
                                                                             train_data['Insult'],
                                                                             train_data['Comment'],
                                                                             train_data['Insult'], ratio=1)

        print('Preprocessing train')
        train_data['Comment'] = data_preprocessing(train_data['Comment'])
        print('Preprocessing test')
        test_data['Comment'] = data_preprocessing(test_data['Comment'])

        print('Writing input files for fasttext')
        write_input_fasttext_cls(train_data, os.path.join(dpath, 'train'), 'train')
        write_input_fasttext_cls(test_data, os.path.join(dpath, 'test'), 'test')

        write_input_fasttext_emb(train_data, os.path.join(dpath, 'train'), 'train')
        write_input_fasttext_emb(test_data, os.path.join(dpath, 'test'), 'test')

        print('Writing input normalized input files')
        train_data.to_csv(os.path.join(dpath, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(dpath, 'test.csv'), index=False)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
