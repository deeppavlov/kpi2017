import os
import copy
import numpy as np
import subprocess
import urllib.request


class EmbeddingsDict(object):
    def __init__(self, opt, embedding_dim):
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.opt = copy.deepcopy(opt)
        self.load_items()

        self.fasttext_path = os.path.join(os.path.expanduser(self.opt.get('fasttext_dir', '')), 'fasttext')
        if not self.opt.get('fasttext_dir') or not os.path.isfile(self.fasttext_path):
            raise RuntimeError('There is no fasttext executable provided.')
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

    def add_items(self, sentence_li):

        command = [self.fasttext_path, 'print-word-vectors', self.fasttext_model_file]
        unk_tokens = []
        for sen in sentence_li:
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            for tok in tokens:
                if self.tok2emb.get(tok) is None:
                    unk_tokens.append(tok)
        if len(unk_tokens) > 0:
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            tok_string = ('\n'.join(unk_tokens)).encode()
            stdout = p.communicate(input=tok_string)[0]
            stdout_li = stdout.decode().split('\n')[:-1]
            for line in stdout_li:
                values = line.rsplit(sep=' ', maxsplit=self.embedding_dim + 1)
                word = values[0]
                coefs = np.asarray(values[1:-1], dtype='float32')
                self.tok2emb[word] = coefs

    def save_items(self, fname):
        f = open(fname + '.emb', 'w')
        string = '\n'.join([el[0] + ' ' + self.emb2str(el[1]) for el in self.tok2emb.items()])
        f.write(string)
        f.close()

    def emb2str(self, vec):
        string = ' '.join([str(el) for el in vec.tolist()])
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