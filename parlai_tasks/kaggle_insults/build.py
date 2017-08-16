import parlai.core.build_data as build_data
import os
import re
import string
try:
    import pandas as pd
except ImportError:
    raise ImportError('Could not initialize Pandas library. Please, ensure that it\'s installed.')


def data_preprocessing(data):
    max_sym_len = 0
    max_word_len = 0
    lens_in_words = []
    lens_in_symbs = []

    for i, comment in enumerate(data['Comment']):
        data.iloc[i, 1] = data.iloc[i, 1].lower()
        data.iloc[i, 1] = re.sub('n\'t', ' not', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub('\'m', ' ', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub('\'s', ' ', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub('\'re', ' ', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub('\'ve', ' ', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub('\'d', ' ', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub(' im ', ' i', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub(' ur ', ' you ', data.iloc[i, 1])
        # data.iloc[i, 1] = re.sub('\? ', ' ', data.iloc[i, 1])
        # data.iloc[i, 1] = re.sub('[!?]+', ' mysignssymbol ', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub(r'href=[\'"]?([^\'" >]+)', ' ', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub('[^a-z]+', ' ',
                                 data.iloc[i, 1])  # replace everything not lowercase literals with space
        data.iloc[i, 1] = re.sub('\s+', ' ', data.iloc[i, 1]).strip()  # replace multiple spaces
        for letter in string.ascii_lowercase:  # replace multiple letters (3 and more)
            data.iloc[i, 1] = re.sub(letter * 3 + '+', letter, data.iloc[i, 1]).strip()
            # data.iloc[i, 1] = re.sub('mysignssymbol', '<SIGNS>', data.iloc[i, 1])
        data.iloc[i, 1] = re.sub('\s+', ' ', data.iloc[i, 1]).strip()  # replace multiple spaces
        if max_sym_len < len(comment):
            max_sym_len = len(comment)
        if max_word_len < len(comment.split(' ')):
            max_word_len = len(comment.split(' '))
        lens_in_words.append(len(comment.split(' ')))
        lens_in_symbs.append(len(comment))
    return data


def write_input_file_fasttext(data, path, data_name):
    f = open(path + '_character.txt', 'w')
    for i in range(data.shape[0]):
        if data_name == 'train' or data_name == 'valid':
            f.write('__label__' + str(data.iloc[i, 0]) + " " + data.iloc[i, 1] + '\n')
        elif data_name == 'test':
            f.write(data.iloc[i, 1] + '\n')
        else:
            print('Incorrect data name')
    f.close()


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'kaggle_insults')
    # define version if any
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if not opt.get('raw_dataset_path'):
            raise RuntimeError('Please download dataset files from'
                               ' https://www.kaggle.com/c/detecting-insults-in-social-commentary/data'
                               ' and set path to their directory in raw-dataset-path parameter')
        raw_path = os.path.abspath(opt['raw_dataset_path'])
        train_file = os.path.join(raw_path, 'train.csv')
        valid_file = os.path.join(raw_path, 'test_with_solutions.csv')
        test_file = os.path.join(raw_path, 'impermium_verification_labels.csv')
        if not os.path.isfile(train_file) or not os.path.isfile(valid_file) or not os.path.isfile(test_file):
            raise RuntimeError('Please download dataset files from'
                               ' https://www.kaggle.com/c/detecting-insults-in-social-commentary/data'
                               ' and set path to their directory in raw-dataset-path parameter')


        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        train_data = pd.read_csv(train_file)
        train_data = train_data.drop('Date', axis=1)

        test_data = pd.read_csv(test_file)
        test_data = test_data.drop('id', axis=1)
        test_data = test_data.drop('Usage', axis=1)
        test_data = test_data.drop('Date', axis=1)

        valid_data = pd.read_csv(valid_file)
        valid_data = valid_data.drop('Date', axis=1)
        valid_data = valid_data.drop('Usage', axis=1)

        print('Preprocessing train')
        train_data = data_preprocessing(train_data)
        print('Preprocessing test')
        test_data = data_preprocessing(test_data)
        print('Preprocessing valid')
        valid_data = data_preprocessing(valid_data)

        print('writing input file for train')
        write_input_file_fasttext(train_data, os.path.join(dpath, 'train'), 'train')
        print('writing input file for valid')
        write_input_file_fasttext(valid_data, os.path.join(dpath, 'valid'), 'valid')
        print('writing input file for test')
        write_input_file_fasttext(test_data, os.path.join(dpath, 'test'), 'test')

        train_data.to_csv(os.path.join(dpath, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(dpath, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(dpath, 'test.csv'), index=False)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
