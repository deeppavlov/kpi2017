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


import parlai.core.build_data as build_data
import os
import xml.etree.ElementTree as ET
import csv


def clean_dataset(path):
    """Remove duplicates from the dataset and write clean data in .tsv files

    Args:
        path: a path to the dataset
    """

    with open(path, 'r') as labels_file:
        context = ET.iterparse(labels_file, events=("start", "end"))
        # turn it into an iterator
        context = iter(context)
        # get the root element
        event, root = next(context)

        with open(os.path.splitext(path)[0] + '.tsv', 'w') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')

            same_set = set()

            for event, elem in context:
                if event == "end" and elem.tag == "paraphrase":
                    question = []
                    y = None
                    for child in elem.iter():
                        if child.get('name') == 'text_1':
                            question.append(child.text)
                        if child.get('name') == 'text_2':
                            question.append(child.text)
                        if child.get('name') == 'class':
                            y = 1 if int(child.text) >= 0 else 0
                    root.clear()
                    check_string = "\n".join(question)
                    if check_string not in same_set:
                        writer.writerow([y, question[0], question[1]])
                        same_set.add(check_string)



def build(opt, shared):
    """Set up full path to datasets files.

    Args:
        opt: given parameters
    """

    # define version if any
    version = '1.1'

    # get path to data directory
    if (opt['raw_dataset_path'] is not None
        and os.path.isdir(opt['raw_dataset_path'])
        and build_data.built(opt['raw_dataset_path'], version_string=version)):
            if shared is None:
                print('Setting the raw_dataset_path parameter for datasets.')
            dpath = opt['raw_dataset_path']
    else:
        if shared is None:
            print('The raw_dataset_path parameter is not set or it is invalid.'
                  ' Setting the datapath parameter for datasets.')
        dpath = os.path.join(opt['datapath'], 'paraphrases')

        # check if data had been previously built
        if not build_data.built(dpath, version_string=version):
            print('[building data: ' + dpath + ']')

            # make a clean directory if needed
            if build_data.built(dpath):
                # an older version exists, so remove these outdated files.
                build_data.remove_dir(dpath)
            build_data.make_dir(dpath)

            # download the data.
            url = 'http://paraphraser.ru/download/get?file_id='  # datasets URL

            fname = 'paraphraser.zip'
            build_data.download(url+'1', dpath, fname)
            # uncompress it
            build_data.untar(dpath, fname)
            path = os.path.join(dpath, 'paraphrases.xml')
            clean_dataset(path)

            fname = 'paraphraser_gold.zip'
            build_data.download(url+'5', dpath, fname)
            # uncompress it
            build_data.untar(dpath, fname)
            path = os.path.join(dpath, 'paraphrases_gold.xml')
            clean_dataset(path)

            # mark the data as built
            build_data.mark_done(dpath, version_string=version)

    datafile = set_path(opt, dpath)
    return datafile


def set_path(opt, dpath):
    """Join the path to datasets directory with datasets file names and return result."""

    dt = opt['datatype'].split(':')[0]
    fname = 'paraphrases'
    if dt == 'test':
        fname += '_gold'
    fname += '.tsv'
    datafile = os.path.join(dpath, fname)
    return datafile


