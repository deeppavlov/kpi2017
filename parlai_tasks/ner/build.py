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
