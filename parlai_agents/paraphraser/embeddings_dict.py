import os
import copy
import numpy as np
import subprocess


class EmbeddingsDict(object):
    def __init__(self, opt):
        self.tok2emb = {}
        self.opt = copy.deepcopy(opt)
        self.load_items()

    def add_items(self, sentence_li):
        fasttext_model = os.path.join(self.opt['datapath'], 'paraphrases', self.opt.get('fasttext_model'))
        fasttext_run = os.path.join(self.opt['fasttext_dir'], 'fasttext')
        if not os.path.isfile(fasttext_model) or not os.path.isfile(fasttext_model):
            print('Error. There is no fasttext executable file or fasttext trained model provided.')
            exit()
        else:
            command = [fasttext_run, 'print-word-vectors', fasttext_model]
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
                values = line.rsplit(sep=' ', maxsplit=self.opt['embedding_dim'] + 1)
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
                    values = line.rsplit(sep=' ', maxsplit=self.opt['embedding_dim'])
                    assert(len(values) == self.opt['embedding_dim'] + 1)
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.tok2emb[word] = coefs