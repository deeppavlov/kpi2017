import copy

import dynet as dy
import numpy as np
from .dictionary import POSDictionaryAgent


class POSTagger:

    def __init__(self, opt, word_dict):
        super(POSTagger, self).__init__()

        self.opt = copy.deepcopy(opt)
        self.word_dict = word_dict  # type: POSDictionaryAgent
        self.opt['tags_size'] = len(self.word_dict.labels_dict)

        self.params = dy.ParameterCollection()

        self.pWs = self.params.add_parameters((opt['hidden_size'], opt['hidden_size']))
        self.pbs = self.params.add_parameters(opt['hidden_size'])

        self.pWn = self.params.add_parameters((opt['hidden_size'], opt['hidden_size']))
        self.pbn = self.params.add_parameters(opt['hidden_size'])

        self.pWx = self.params.add_parameters((opt['hidden_size'], opt['word_dim']))
        self.pbx = self.params.add_parameters(opt['hidden_size'])

        self.pWy = self.params.add_parameters((self.opt['tags_size'], opt['hidden_size']))
        self.pby = self.params.add_parameters(self.opt['tags_size'])

        dy.renew_cg()

        self.trainer = dy.SimpleSGDTrainer(self.params, learning_rate=self.opt['learning_rate'])

    def word_char_emb(self, str_word):
        assert isinstance(str_word, str)
        w = ' ' + str_word + ' '
        wl = len(w)
        ec = self.opt['word_dim']
        emb = np.zeros(ec)
        n = 3
        for i in range(wl - n + 1):
            emb[hash(w[i:i+n+1]) % ec] = 1
        return emb

    def train_single(self, x, y):
        Ws = dy.parameter(self.pWs)
        Wn = dy.parameter(self.pWn)
        Wx = dy.parameter(self.pWx)
        b = dy.parameter(self.pbx)

        Wy = dy.parameter(self.pWy)
        by = dy.parameter(self.pby)

        s = [(self.word_dict.start_token, self.word_dict.labels_dict.start_token)] +\
            list(zip(x, y)) +\
            [(self.word_dict.end_token, self.word_dict.labels_dict.end_token)]
        x_ = np.array([self.word_char_emb(t[0]) for t in s], 'float32').T
        y_ = np.array([self.word_dict.labels_dict[d[1]] for d in s], dtype='int')
        X = dy.reshape(dy.inputTensor(x_), x_.shape[0:1], len(y_))

        h = dy.zeros(self.opt['hidden_size'], batch_size=len(y_))
        for t in range(self.opt['depth']):
            new_h = []
            for i in range(len(y_)):
                neighs = list(range(max(0, i-1), min(len(y_), i+2)))
                new_h.append(dy.tanh(Ws * dy.pick_batch_elem(h, i) + Wx * dy.pick_batch_elem(X, i) +
                                     Wn * dy.sum_batches(dy.pick_batch_elems(h, neighs)) + b))
            h = dy.concatenate_to_batch(new_h)
        y_hat = dy.softmax(Wy * h + by)
        probs = dy.pick_batch(y_hat, y_)
        loss = dy.mean_batches(-dy.log(probs))
        return y_hat, loss

    def train(self, batch):
        dy.renew_cg()
        losses = []
        probs = []
        for x, y in batch:
            prob, loss = self.train_single(x, y)
            probs.append(prob)
            losses.append(loss)
        losses = dy.average(losses)
        losses.forward()
        res = []
        for prob in probs:
            prob = prob.npvalue()
            res.append(prob[:, 1:-1].argmax(0))
        losses.backward()
        self.trainer.update()
        return losses.npvalue(), res

    def forward_single(self, x):
        Ws = dy.parameter(self.pWs)
        Wn = dy.parameter(self.pWn)
        Wx = dy.parameter(self.pWx)
        b = dy.parameter(self.pbx)

        Wy = dy.parameter(self.pWy)
        by = dy.parameter(self.pby)

        words = [self.word_dict.start_token] + list(x) + [self.word_dict.end_token]
        x_ = np.array([self.word_char_emb(w) for w in words], 'float32').T
        X = dy.reshape(dy.inputTensor(x_), x_.shape[0:1], len(words))

        h = dy.zeros(self.opt['hidden_size'], batch_size=len(words))
        for t in range(self.opt['depth']):
            new_h = []
            for i in range(len(words)):
                neighs = list(range(max(0, i-1), min(len(words), i+2)))
                new_h.append(dy.tanh(Ws * dy.pick_batch_elem(h, i) + Wx * dy.pick_batch_elem(X, i) +
                                     Wn * dy.sum_batches(dy.pick_batch_elems(h, neighs)) + b))
            h = dy.concatenate_to_batch(new_h)
        y_hat = dy.softmax(Wy * h + by)
        return y_hat

    def forward(self, batch):
        dy.renew_cg()
        probs = []
        for x, _ in batch:
            probs.append(self.forward_single(x))
        dy.concatenate_to_batch(probs).forward()
        res = []
        for prob in probs:
            prob = prob.npvalue()
            res.append(prob[:, 1:-1].argmax(0))
        return res
