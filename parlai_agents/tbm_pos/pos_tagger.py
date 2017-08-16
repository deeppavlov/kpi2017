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

    def __init__(self, opt, word_dict, state_dict=None, word_embeddings_size=50, word_window_size=2,
                 pos_embedding_size=15, prev_pos_count=3):
        super(POSTagger, self).__init__()

        self.word_embedding_size = word_embeddings_size
        self.word_windows_size = word_window_size
        self.prev_pos_count = prev_pos_count
        self.pos_embedding_size = pos_embedding_size

        self.opt = copy.deepcopy(opt)
        self.opt['cuda'] = self.opt.get('cuda') and torch.cuda.is_available()
        self.word_dict = word_dict  # type: POSDictionaryAgent

        self.word_emb = self.gpu(nn.Embedding(len(word_dict), self.word_embedding_size))
        self.pos_emb = self.gpu(nn.Embedding(len(word_dict.labels_dict), self.pos_embedding_size))
        words_count = self.word_windows_size*2 + 1
        self.linear = self.gpu(nn.Linear(self.word_embedding_size*words_count +
                                         self.pos_embedding_size*self.prev_pos_count,
                                         len(word_dict.labels_dict)))
        self.softmax = self.gpu(nn.Softmax())

        if state_dict:
            self.load_state_dict(state_dict)

    def zero_var(self):
        return self.gpu(Variable(torch.FloatTensor([0])))

    def set_input(self, input_seq):
        # input_seq = input_seq.split(' ')
        # words_ids = [self.word_dict.__getitem__(word) for word in input_seq]
        start = self.word_dict[self.word_dict.start_token]
        end = self.word_dict[self.word_dict.end_token]
        words_ids = [start]*self.word_windows_size + self.word_dict.txt2vec(input_seq) + [end]*self.word_windows_size
        words_count = len(words_ids) - self.word_windows_size*2
        words = self.word_emb(self.gpu(Variable(torch.LongTensor(words_ids))))
        start = [self.word_dict.labels_dict[self.word_dict.labels_dict.start_token]]*self.prev_pos_count
        return POS_Tagger_State(words, 0, words_count, start, None,
                                words_count == 0)

    def act(self, state=None, action=None, output=None):
        if state is None:
            raise RuntimeError('Starting state is generated in set_input')
            # start = self.word_dict.labels_dict.__getitem__(self.word_dict.labels_dict.start_token)
            # return POS_Tagger_State(0, start, None, False)
        assert action is not None
        assert output is not None
        terminated = state.input_index + 1 >= state.words_count
        return POS_Tagger_State(state.words, state.input_index + 1, state.words_count, (state.prev_pos + [action])[-self.prev_pos_count:], output, terminated)

    def embed_states(self, states):
        words = []
        prev_pos = []
        for state in states:
            words.append(state.words[state.input_index:state.input_index+self.word_windows_size*2+1].view(1, -1))
            prev_pos.append(state.prev_pos)
        word_embeddings = torch.cat(words, dim=0)
        prev_pos = self.gpu(Variable(torch.LongTensor(prev_pos)))
        return word_embeddings, self.pos_emb(prev_pos).view(len(states), -1)

    def forward(self, states):
        word_embeddings, pos_embeddings = self.embed_states(states)
        x = torch.cat([word_embeddings, pos_embeddings], dim=1)
        scores = self.linear(x)
        return scores

    def calculate_gold_path(self, seq):
        words, pos_tags = zip(*seq)
        gold_actions = [self.word_dict.labels_dict.__getitem__(p) for p in pos_tags]
        return words, gold_actions
