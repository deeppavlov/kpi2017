import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'UD')
    version = 'None'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        dev_fname = 'en-ud-dev.conllu'
        train_fname = 'en-ud-train.conllu'
        stats_fname = 'stats.xml'
        url = 'https://github.com/UniversalDependencies/UD_English/raw/master/'
        build_data.download(url+dev_fname, dpath, dev_fname)
        build_data.download(url+train_fname, dpath, train_fname)
        build_data.download(url+stats_fname, dpath, stats_fname)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
