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

def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'gareev')
    # define version if any
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # assamble the data.
        iob_files = glob(os.path.join('dataset/gareev/', '*.iob'))
        with open(os.path.join(dpath, 'gareev.txt'), 'w') as outfile:
            outfile.write('-DOCSTART- -X- -X- O\n')
            for iob in iob_files:
                with open(iob) as infile:
                    for line in infile:
                        outfile.write(line)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
