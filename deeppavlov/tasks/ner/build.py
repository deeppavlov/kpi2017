import parlai.core.build_data as build_data
import os
from glob import glob


def build(opt):
    # get path to data directory
    dpath = opt['datapath']
    raw_data_path = opt.get('raw_data_path')
    # define version if any
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[source data path: ' + raw_data_path + ']')
        print('[target data path: ' + dpath + ']')
        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        if not raw_data_path:
            raise RuntimeError('No path for raw dataset specified. Please specify it with ' +
                               '--raw-datapath {path}.')
        # assemble the data.
        iob_files = glob(os.path.join(raw_data_path, '*.iob'))

        if len(iob_files) != 97:
            raise RuntimeError('There must be 97 .iob files. To get the corpus please contact authors'
                          ' of https://link.springer.com/chapter/10.1007/978-3-642-37247-6_27')
        with open(os.path.join(dpath, 'heap.txt'), 'w') as outfile:
            for iob in iob_files:
                with open(iob) as infile:
                    for line in infile:
                        outfile.write(line)
                    outfile.write('\n')
        # mark the data as built
        build_data.mark_done(dpath, version_string=version)


if __name__ == '__main__':
    opt = {'datapath': '/tmp/gareev'}
    opt['raw_data_path'] = '/home/mikhail/Data/gareev'
    build(opt)
