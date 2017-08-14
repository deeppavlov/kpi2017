from .dictionary import POSDictionaryAgent
from .pos_tagger import POSTagger

import torch
import torch.nn as nn


class POSTaggerModel(object):

    def __init__(self, opt, word_dict, state_dict=None):
        self.opt = opt

        self.train_func = self.train_naive
        self.forward = self.forward_naive

        self.word_dict = word_dict  # type: POSDictionaryAgent
        self.state_dict = state_dict
        self.network = POSTagger(opt, word_dict)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])
        parameters = [p for p in self.network.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(parameters, self.opt['learning_rate'])
        if state_dict:
            new_state = set(self.optimizer.state_dict().keys())
            for k in list(state_dict['optimizer'].keys()):
                if k not in new_state:
                    del state_dict['optimizer'][k]
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def train_naive(self, input_seq, gold_path):
        state = self.network.set_input(input_seq)
        path = []
        loss = self.network.zero_loss()
        for x, y in zip(input_seq, gold_path):
            output = self.network.forward(state)
            best_action = output.max(1)[1].data[0][0]
            path.append(best_action)
            softmax_output = nn.LogSoftmax()(output)
            loss = loss - softmax_output[0, y]
            state = self.network.act(state, y, output)

        loss = loss / len(input_seq)
        # loss.backward()
        return loss, path

    def forward_naive(self, input_seq):
        state = self.network.set_input(input_seq)
        path = []
        while True:
            output = self.network.forward(state)
            best_action = output.max(1)[1].data[0][0]
            path.append(best_action)
            state = self.network.act(state, best_action, output)
            if state.terminated:
                break

        return path

    def train_batch(self, batch, *args, **kwargs):
        self.optimizer.zero_grad()
        res = [self.train_func(*args, **kwargs, input_seq=x, gold_path=y) for x, y in batch]
        losses, paths = zip(*res)
        batch_loss = torch.cat(losses).mean()
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss, paths

    def save(self, fname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            'word_dict': self.word_dict,
            'config': self.opt,
        }
        try:
            torch.save(params, fname)
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')
