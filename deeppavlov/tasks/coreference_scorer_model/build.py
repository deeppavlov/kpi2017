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
from os.path import join
import time
from . import utils
from ...utils import coreference_utils


def build(opt):
    """prepares datasets and other dependencies for CoreferenceTeacher"""

    # get path to data directory and create folders tree
    dpath = join(opt['datapath'])
    # define version if any, and languages
    version = '1.0'
    language = opt['language']
    dpath = join(dpath, 'coreference_scorer_model', language)
    build_data.make_dir(dpath)
    build_data.make_dir(join(dpath, opt['predictions_folder'], 'train'))
    build_data.make_dir(join(dpath, opt['predictions_folder'], 'test'))
    build_data.make_dir(join(dpath, opt['predictions_folder'], 'valid'))
    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + '] ...')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)

        # Build the folders tree
        build_data.make_dir(dpath)
        build_data.make_dir(join(dpath, 'report', 'response_files'))
        build_data.make_dir(join(dpath, 'report', 'results'))
        build_data.make_dir(join(dpath, 'scorer'))
        build_data.make_dir(join(dpath, 'train'))
        build_data.make_dir(join(dpath, 'test'))
        build_data.make_dir(join(dpath, 'valid'))

        # urls
        dataset_url = 'http://rucoref.maimbava.net/files/rucoref_29.10.2015.zip'
        scorer_url = 'http://conll.cemantix.org/download/reference-coreference-scorers.v8.01.tar.gz'

        # download the conll-2012 scorer v 8.1
        start = time.time()

        print('[Downloading the conll-2012 scorer]...')
        build_data.download(scorer_url, join(dpath, 'scorer'), 'reference-coreference-scorers.v8.01.tar.gz')
        build_data.untar(join(dpath, 'scorer'), 'reference-coreference-scorers.v8.01.tar.gz')
        print('[Scorer was downloaded]...')

        fname = 'rucoref_29.10.2015.zip'
        if not os.path.isdir(join(dpath, 'rucoref_29.10.2015')):
            print('[Downloading the rucoref dataset]...')
            build_data.make_dir(join(dpath, 'rucoref_29.10.2015'))
            build_data.download(dataset_url, join(dpath, 'rucoref_29.10.2015'), fname)
            # uncompress it
            build_data.untar(join(dpath, 'rucoref_29.10.2015'), 'rucoref_29.10.2015.zip')
            print('End of downloading: took {0:.3f}s'.format(time.time() - start))

        # Convertation rucorpus files in conll files
        conllpath = join(dpath, 'ru_conll')
        build_data.make_dir(conllpath)
        coreference_utils.RuCoref2CoNLL(
            join(dpath, 'rucoref_29.10.2015'), conllpath, language)

        # splits conll files
        start = time.time()
        conlls = join(dpath, 'ru_conlls')
        build_data.make_dir(conlls)
        coreference_utils.split_doc(join(conllpath, language +
                                         '.v4_conll'), conlls, language)
        build_data.remove_dir(conllpath)

        # create train valid test partitions
        utils.train_valid_test_split(conlls, join(dpath, 'train'),
                                     join(dpath, 'valid'), join(dpath, 'test'),
                                     valid_ratio=opt['valid_ratio'], test_ratio=opt['test_ratio'],
                                     seed=opt['teacher_seed'])

        build_data.remove_dir(conlls)
        print('End of data splitting: took {0:.3f}'.format(
            time.time() - start))

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
        print('[Datasets done.]')
        return None
