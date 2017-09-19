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
from glob import glob
import ftplib
import numpy as np
import random


def is_end_of_sentence(prev_token, current_token):
    is_capital = current_token[0].isupper()
    is_punctuation = prev_token in ('!', '?', '.')
    return is_capital and is_punctuation


def ftp_gareev_loader(dpath, heap_filename='heap.txt'):
    print('Downloading data from FTP server...')
    server = 'share.ipavlov.mipt.ru'
    username = 'anonymous'
    password = ''
    directory = '/datasets/gareev/'
    filematch = '*.iob'
    ftp = ftplib.FTP(server)
    ftp.login(username, password)
    ftp.cwd(directory)

    prev_token = '\n'
    if not os.path.exists(dpath):
        os.mkdir(dpath)
    tmp_file_path = os.path.join(dpath, 'ner_tmp_file.txt')
    with open(os.path.join(dpath, heap_filename), 'w') as outfile:
        for file_name in ftp.nlst(filematch):
            with open(tmp_file_path, 'wb') as tmp_f:
                ftp.retrbinary('RETR ' + file_name, tmp_f.write)
            with open(tmp_file_path) as tmp_f:
                lines_list = tmp_f.readlines()
            for line in lines_list:
                if len(line) > 2:
                    token, tag = line.split()
                    if not is_end_of_sentence(prev_token, token):
                        outfile.write(token + ' ' + tag + '\n')
                    else:
                        outfile.write('\n' + token + ' ' + tag + '\n')
                    prev_token = token


def build(opt):
    version = '1.1'
    dpath = os.path.join(opt['datapath'], 'ner')

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[target data path: ' + dpath + ']')
        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        ftp_gareev_loader(dpath)
        # mark the data as built
        build_data.mark_done(dpath, version_string=version)