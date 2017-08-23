import parlai.core.build_data as build_data
import os
import re
import string
import _pickle as cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
try:
    import pandas as pd
except ImportError:
    raise ImportError('Could not initialize Pandas library. Please, ensure that it\'s installed.')

def data_preprocessing(f, path):
    f = [x.lower() for x in f]
    f = [x[1:-1] for x in f]
    f = [x.replace("\\n", " ") for x in f]
    f = [x.replace("\\t", " ") for x in f]
    f = [x.replace("\\xa0", " ") for x in f]
    f = [x.replace("\\xc2", " ") for x in f]

    f = [re.sub('[!!]+', ' !! ', x) for x in f]
    f = [re.sub('!', ' ! ', x) for x in f]
    f = [re.sub('! !', '!!', x) for x in f]

    f = [re.sub('[\?\?]+', ' ?? ', x) for x in f]
    f = [re.sub('\?', ' ? ', x) for x in f]
    f = [re.sub('\? \?', '??', x) for x in f]

    f = [re.sub('[\?!]+', ' ?! ', x) for x in f]

    f = [re.sub('[\.\.]+', '..', x) for x in f]
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

    bad_words_file = open(path + "/badwords.txt", "r")
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
    f = open(data_name + '_fasttext_emb.txt', 'w')
    for i in range(data.shape[0]):
        if data_name == 'train' or data_name == 'test':
            f.write(data.iloc[i,1] + '\n')
        else:
            print('Incorrect data name')
    f.close()

def balance_dataset(dataset_0, labels_0, dataset_1, labels_1, ratio=1):
    initial_train_size = dataset_0.shape[0]
    insult_inds = np.nonzero(labels_1)[0]
    num_insults_0 = len(np.nonzero(labels_0)[0])
    num_insults_1 = len(np.nonzero(labels_1)[0])
    insult_inds_to_add = insult_inds[np.random.randint(low=0, high=num_insults_1,
                                                       size=(ratio * (initial_train_size - num_insults_0) - num_insults_0))]
    result = dataset_0.append(dataset_1.iloc[insult_inds_to_add])
    result_labels = labels_0.append(labels_1.iloc[insult_inds_to_add])
    return result, result_labels


def ngrams_selection(train_data, train_labels, ind, dpath,
                     ngram_range_=(1, 1), max_num_features=100,
                     analyzer_type='word'):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range_, sublinear_tf=True, analyzer=analyzer_type)

    X_train = vectorizer.fit_transform(train_data)

    if max_num_features < X_train.shape[1]:
        ch2 = SelectKBest(chi2, k=max_num_features)
        ch2.fit(X_train, train_labels)
        data_struct = {'vectorizer': vectorizer, 'selector': ch2}
        print ('creating ', os.path.join(dpath, 'ngrams_vect_' + ind + '.bin'))
        with open(os.path.join(dpath, 'ngrams_vect_' + ind + '.bin'), 'wb') as f:
            cPickle.dump(data_struct, f)
    else:
        data_struct = {'vectorizer': vectorizer}
        print ('creating', os.path.join(dpath, 'ngrams_vect_' + ind + '.bin'))
        with open(os.path.join(dpath, 'ngrams_vect_' + ind + '.bin'), 'wb') as f:
            cPickle.dump(data_struct, f)
    return

def ngrams_you_are(data):
    g = [x.lower()
         .replace("you are", " SSS ")
         .replace("you're", " SSS ")
         .replace(" ur ", " SSS ")
         .replace(" u ", " SSS ")
         .replace(" you ", " SSS ")
         .replace(" yours ", " SSS ")
         .replace(" u r ", " SSS ")
         .replace(" are you ", " SSS ")
         .replace(" urs ", " SSS ")
         .replace(" r u ", " SSS ").split("SSS")[1:]
         for x in data]
    f = []
    for x in g:
        fts = " "
        for y in x:
            w = y.strip().replace("?",".").split(".")
            fts = fts + " " + w[0]
        f.append(fts)
    return f

def create_vectorizer_selector(train_data, train_labels, dpath,
                               ngram_list=[1], max_num_features_list=[100], analyzer_type_list=['word']):

    for i in range(len(ngram_list)):
        ngrams_selection(train_data, train_labels, 'general_' + str(i), dpath,
                         ngram_range_=(ngram_list[i], ngram_list[i]),
                         max_num_features=max_num_features_list[i],
                         analyzer_type=analyzer_type_list[i])
    you_are_data = ngrams_you_are(train_data)
    ngrams_selection(you_are_data, train_labels, 'special', dpath,
                     ngram_range_=(1,1), max_num_features=100)
    return

def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'insults')
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

        # merge train and valid due to use of cross validation
        train_data = train_data.append(valid_data)

        if opt.get('balance_train_dataset'):
            if opt['balance_train_dataset'] == True:
                train_data['Comment'],train_data['Insult'] = balance_dataset(train_data['Comment'],
                                                                             train_data['Insult'],
                                                                             train_data['Comment'],
                                                                             train_data['Insult'], ratio=1)

        print('Preprocessing train')
        train_data['Comment'] = data_preprocessing(train_data['Comment'], raw_path)
        print('Preprocessing test')
        test_data['Comment'] = data_preprocessing(test_data['Comment'], raw_path)
        valid_data['Comment'] = data_preprocessing(valid_data['Comment'], raw_path)

        create_vectorizer_selector(train_data['Comment'], train_data['Insult'], dpath,
                                   ngram_list=[1, 2, 3, 4, 5, 3],
                                   max_num_features_list=[2000, 4000, 100, 1000, 1000, 2000],
                                   analyzer_type_list=['word', 'word', 'word', 'char', 'char', 'char'])

        print('Writing input files for fasttext')
        write_input_fasttext_cls(train_data, os.path.join(dpath, 'train'), 'train')
        write_input_fasttext_cls(test_data, os.path.join(dpath, 'test'), 'test')

        write_input_fasttext_emb(train_data, os.path.join(dpath, 'train'), 'train')
        write_input_fasttext_emb(test_data, os.path.join(dpath, 'test'), 'test')

        print('Writing input normalized input files')
        train_data.to_csv(os.path.join(dpath, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(dpath, 'test.csv'), index=False)
        valid_data.to_csv(os.path.join(dpath, 'valid.csv'), index=False)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
