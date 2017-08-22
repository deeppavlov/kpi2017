import os
import copy
import numpy as np
import urllib.request
import fasttext


class EmbeddingsDict(object):
    def __init__(self, opt, embedding_dim):
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.opt = copy.deepcopy(opt)
        self.load_items()

        if not self.opt.get('fasttext_model'):
            raise RuntimeError('No pretrained fasttext model provided')
        self.fasttext_model_file = self.opt.get('fasttext_model')
        if not os.path.isfile(self.fasttext_model_file):
            ftppath = os.environ.get('IPAVLOV_FTP')
            if not ftppath:
                raise RuntimeError('No pretrained fasttext model provided')
            fname = os.path.basename(self.fasttext_model_file)
            try:
                print('Trying to download a pretrained fasttext model from the ftp server')
                urllib.request.urlretrieve(ftppath + '/paraphraser_data/' + fname, self.fasttext_model_file)
                print('Downloaded a fasttext model')
            except:
                raise RuntimeError('Looks like the `IPAVLOV_FTP` variable is set incorrectly')
        self.fasttext_model = fasttext.load_model(self.fasttext_model_file)

    def add_items(self, sentence_li):
        for sen in sentence_li:
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            for tok in tokens:
                if self.tok2emb.get(tok) is None:
                    self.tok2emb[tok] = self.fasttext_model[tok]

    def save_items(self, fname):
        f = open(fname + '.emb', 'w')
        string = '\n'.join([el[0] + ' ' + self.emb2str(el[1]) for el in self.tok2emb.items()])
        f.write(string)
        f.close()

    def emb2str(self, vec):
        string = ' '.join([str(el) for el in vec])
        return string

    def load_items(self):
        """Initialize embeddings from file."""
        if self.opt.get('model_file') is not None:
            fname = self.opt['model_file']
        if self.opt.get('pretrained_model') is not None:
            fname = self.opt['pretrained_model']

        if not os.path.isfile(fname+'.emb'):
            print('There is no %s.emb file provided. Initializing new dictionary.' % fname)
        else:
            print('Loading existing dictionary from %s.emb.' % fname)
            with open(fname+'.emb', 'r') as f:
                for line in f:
                    values = line.rsplit(sep=' ', maxsplit=self.embedding_dim)
                    assert(len(values) == self.embedding_dim + 1)
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.tok2emb[word] = coefs