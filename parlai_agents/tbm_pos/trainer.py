from collections import namedtuple

from .dictionary import POSDictionaryAgent
from .pos_tagger import POSTagger

import torch
import torch.nn as nn

import timeit


State = namedtuple('State', 'inner_state output path score_sum terminated')


class Trainer:
    def train(self, batch):
        raise NotImplemented()

    def forward(self, batch):
        raise NotImplemented()


class NaiveTrainer(Trainer):

    def __init__(self, learning_rate, word_dict, network):
        self.times = []

        self.word_dict = word_dict  # type: POSDictionaryAgent
        self.labels_len = len(self.word_dict.labels_dict)
        self.network = network

        parameters = [p for p in self.network.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(parameters, learning_rate)

    def train(self, batch):
        self.optimizer.zero_grad()
        x, ys = zip(*batch)
        states = [self.network.set_input(input_seq) for input_seq in x]
        paths = [[] for _ in states]
        loss = self.network.zero_var()
        step = 0
        loss_cnt = 0
        while True:
            indexes = [i for i in range(len(states)) if not states[i].terminated]
            if not indexes:
                break
            output = self.network.forward([states[i] for i in indexes])  # 64x21
            output = self.network.softmax(output)
            best_actions = output.max(1)[1].data.squeeze(1).tolist()  # 64
            for i in range(len(best_actions)):
                loss_cnt += 1
                index = indexes[i]
                y = ys[index][step]
                paths[index].append(best_actions[i])
                loss -= output[i, y]
                states[index] = self.network.act(states[index], y, output[i])
            step += 1
        loss /= loss_cnt
        loss.backward()
        self.optimizer.step()
        return loss, paths

    def forward(self, batch):
        x, _ = zip(*batch)
        states = [self.network.set_input(input_seq) for input_seq in x]
        paths = [[] for _ in states]
        while True:
            indexes = [i for i in range(len(states)) if not states[i].terminated]
            if not indexes:
                break
            output = self.network.forward([states[i] for i in indexes])
            best_actions = output.max(1)[1].data.squeeze(1).tolist()
            for i in range(len(best_actions)):
                index = indexes[i]
                paths[index].append(best_actions[i])
                states[index] = self.network.act(states[index], best_actions[i], output[i])
        return paths


class BeamTrainer(Trainer):

    def __init__(self, learning_rate, beam_size, word_dict, network):
        self.beam_size = beam_size
        self.times = []

        self.word_dict = word_dict  # type: POSDictionaryAgent
        self.labels_len = len(self.word_dict.labels_dict)
        # todo: purge network from self
        self.network = network

        parameters = [p for p in self.network.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(parameters, learning_rate)

    def compute_beam_scores(self, states):
        new_states = []
        # start_time = timeit.default_timer()
        output = self.network.forward([s.inner_state for s in states])
        for i in range(len(states)):
            s = states[i]
            # output = self.network.forward(s.inner_state)
            # todo: sort with torch.sort here
            for a in range(self.labels_len):
                o = output[i, a]
                inner_state = self.network.act(s.inner_state, a, output)
                state = State(inner_state, output[i, :], s.path + [a], s.score_sum + o,
                              inner_state.terminated)
                new_states.append(state)
        # self.times.append(timeit.default_timer() - start_time)
        # print(sum(self.times)/len(self.times))
        return new_states

    def train(self, input_seq, gold_path):
        initial_state = State(inner_state=self.network.set_input(input_seq), output=[], path=[],
                              score_sum=self.network.zero_var(), terminated=False)
        states = [initial_state]
        finished_path = []
        step = 0
        while True:
            step += 1
            states = self.compute_beam_scores(states)
            gold_score = next(state.score_sum for state in states if state.path == gold_path[:step])
            finished_path += [state for state in states if state.terminated is True]
            states = [state for state in states if state.terminated is False]
            if len(states) == 0:
                break
            states.sort(key=lambda state: state.score_sum.data[0], reverse=True)
            states = states[:self.beam_size]
            # todo: allow not to stop when gold_path is in finished, but there are other paths remaining
            if gold_path[:step] not in [state.path for state in states]:
                # print(step, "out of", len(gold_path))
                break

        states += finished_path
        loss = torch.cat([torch.exp(state.score_sum) for state in states]).sum().log() - gold_score
        return loss, None

    def train_batch(self, batch):
        x, ys = zip(*batch)
        beam_size = self.opt['beam_size']
        states = [State(inner_state=self.network.set_input(input_seq), output=[], path=[],
                        score_sum=self.network.zero_var(), terminated=False) for input_seq in x]
        while True:
            indexes = [i for i in range(len(states)) if not states[i].terminated]
            new_states = self.compute_beam_scores([states[i] for i in indexes])
            for i in range(indexes):
                sorted(new_states[i*self.labels_len:(i+1)*self.labels_len], key=lambda state: state.score_sum.data[0],
                       reverse=True)

    def forward(self, input_seq):
        beam_size = self.opt['beam_size']
        initial_state = State(inner_state=self.network.set_input(input_seq), output=[], path=[],
                              score_sum=self.network.zero_var(), terminated=False)
        finished_states = []
        states = [initial_state]
        step = 0
        stop_flag = False
        while not stop_flag:
            step += 1
            states = self.compute_beam_scores(states)
            finished_states += [state for state in states if state.terminated is True]
            states = [state for state in states if state.terminated is False]
            if len(states) == 0:
                # stop_flag = True
                break
            states.sort(key=lambda state: state.score_sum.data[0], reverse=True)
            states = states[:beam_size]
            # allow not to stop when gold_path is in finished, but there are other paths remaining

        states = finished_states
        states.sort(key=lambda state: state.score_sum.data[0], reverse=True)
        states = states[:beam_size]

        return states[0].path

    def train_batch(self, batch, *args, **kwargs):
        self.optimizer.zero_grad()
        res = [self.train_func(*args, **kwargs, input_seq=x, gold_path=y) for x, y in batch]
        losses, paths = zip(*res)
        batch_loss = torch.cat(losses).mean()
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss, paths
