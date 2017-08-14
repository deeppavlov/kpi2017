import copy
import torch.nn as nn
from torch.autograd import Variable

import torch

from collections import namedtuple

# from torch.multiprocessing.pool import Pool
from .dictionary import POSDictionaryAgent

POS_Tagger_State = namedtuple('POS_Tagger_State', 'words input_index words_count prev_pos output terminated')


class POSTagger(nn.Module):

    def gpu(self, obj):
        if self.opt.get('cuda'):
            return obj.cuda(device_id=self.opt.get('gpu'))
        else:
            return obj

    def __init__(self, opt, word_dict, state_dict=None):
        super(POSTagger, self).__init__()
        self.opt = copy.deepcopy(opt)
        self.opt['cuda'] = self.opt.get('cuda') and torch.cuda.is_available()
        self.word_dict = word_dict  # type: POSDictionaryAgent

        self.word_emb = self.gpu(nn.Embedding(len(word_dict), 20))
        self.pos_emb = self.gpu(nn.Embedding(len(word_dict.labels_dict), 5))
        self.linear = self.gpu(nn.Linear(25, len(word_dict.labels_dict)))

        if state_dict:
            self.load_state_dict(state_dict)

    def zero_loss(self):
        return self.gpu(Variable(torch.FloatTensor([0])))

    def set_input(self, input_seq):
        # input_seq = input_seq.split(' ')
        # words_ids = [self.word_dict.__getitem__(word) for word in input_seq]
        words_ids = self.word_dict.txt2vec(input_seq)
        words_count = len(words_ids)
        words = self.word_emb(self.gpu(Variable(torch.LongTensor(words_ids))))
        start = self.word_dict.labels_dict.__getitem__(self.word_dict.labels_dict.start_token)
        return POS_Tagger_State(words, 0, words_count, start, None, words_count == 0)

    def act(self, state=None, action=None, output=None):
        if state is None:
            raise RuntimeError('Starting state is generated in set_input')
            # start = self.word_dict.labels_dict.__getitem__(self.word_dict.labels_dict.start_token)
            # return POS_Tagger_State(0, start, None, False)
        assert action is not None
        assert output is not None
        terminated = state.input_index + 1 >= state.words_count
        return POS_Tagger_State(state.words, state.input_index + 1, state.words_count, action, output, terminated)

    def forward(self, state):
        assert state.words, 'Supply input data!'
        word = state.words[state.input_index]
        pos_embedding = self.pos_emb(self.gpu(Variable(torch.LongTensor([state.prev_pos]))))
        x_vec = torch.cat([word.unsqueeze(0), pos_embedding], dim=1)
        output = self.linear(x_vec)
        return output

    def calculate_gold_path(self, seq):
        words, pos_tags = zip(*seq)
        gold_actions = [self.word_dict.labels_dict.__getitem__(p) for p in pos_tags]
        return words, gold_actions
