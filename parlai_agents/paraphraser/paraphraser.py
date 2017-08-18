import copy
import subprocess
import numpy as np
import os

from . import config
from .model import ParaphraserModel
from .utils import load_embeddings

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str


class ParaphraserDictionaryAgent(DictionaryAgent):

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--dict_class', default=class2str(ParaphraserDictionaryAgent)
        )

    def act(self):
        """Add only words passed in the 'text' field of the observation to this dictionary."""
        text = self.observation.get('text')
        if text:
            self.add_to_dict(self.tokenize(text))
        return {'id': 'ParaphraserDictionary'}

    def tokenize(self, text, building=False):
        """Returns a sequence of tokens from the iterable."""
        questions = (' ').join(text.split('\n')[1:])
        return questions.split(' ')

class ParaphraserAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        ParaphraserAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return ParaphraserDictionaryAgent

    def __init__(self, opt, shared=None):
        self.id = 'ParaphraserAgent'
        self.episode_done = True
        super().__init__(opt, shared)
        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False

        print('create model')
        self.model = ParaphraserModel(opt)
        self.n_examples = 0

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):

        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]
        examples = [self._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batch_size) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]
        batch = self._batchify(examples)

        if 'labels' in observations[0] and not self.opt.get('pretrained_model'):
            self.n_examples += len(examples)
            self.model.update(batch)
        else:
            batch, _ = batch
            predictions = self.model.predict(batch)
            predictions = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]

        return batch_reply

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save(fname)

    def _build_ex(self, ex):
        if 'text' not in ex:
            return

        """Find the token span of the answer in the context for this example.
        """
        inputs = dict()
        texts = ex['text'].split('\n')
        inputs['question1'] = texts[1]
        inputs['question2'] = texts[2]
        if 'labels' in ex:
            inputs['labels'] = ex['labels']

        return inputs

    def _batchify(self, batch):
        question1 = []
        question2 = []
        for ex in batch:
            question1.append(ex['question1'])
            question2.append(ex['question2'])
        self._create_embeddings(question1)
        self._create_embeddings(question2)
        b1 = self._create_batch(question1)
        b2 = self._create_batch(question2)

        if len(batch[0]) == 3:
            y = [1 if ex['labels'][0] == 'Да' else 0 for ex in batch]
            return [b1, b2], y
        else:
            return [b1, b2], None

    def _create_embeddings(self, sentence_li):
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
                if self.model.tok2emb.get(tok) is None:
                    unk_tokens.append(tok)
        if len(unk_tokens) > 0:
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            tok_string = ('\n'.join(unk_tokens)).encode()
            stdout = p.communicate(input=tok_string)[0]
            stdout_li = stdout.decode().split('\n')[:-1]
            for line in stdout_li:
                values = line.rsplit(sep=' ', maxsplit=self.model.embedding_dim + 1)
                word = values[0]
                coefs = np.asarray(values[1:-1], dtype='float32')
                self.model.tok2emb[word] = coefs

    def _create_batch(self, sentence_li):
        embeddings_batch = []
        for sen in sentence_li:
            embeddings = []
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            if len(tokens) >= self.model.max_sequence_length:
                for tok in tokens[len(tokens)-self.model.max_sequence_length:]:
                    emb = self.model.tok2emb.get(tok)
                    if emb is None:
                        print('Error!')
                        exit()
                    embeddings.append(emb)
            else:
                for tok in tokens:
                    emb = self.model.tok2emb.get(tok)
                    if emb is None:
                        print('Error!')
                        exit()
                    embeddings.append(emb)
                    pads = []
                for _ in range(self.model.max_sequence_length - len(tokens)):
                    pads.append(np.zeros(self.model.embedding_dim))
                embeddings = pads + embeddings
            embeddings = np.asarray(embeddings)
            embeddings_batch.append(embeddings)
        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def _predictions2text(self, predictions):
        y = ['Да' if ex > 0.5 else 'Нет' for ex in predictions]
        return y

    def report(self):
        return (
            '[train] updates = %d | exs = %d | loss = %.4f | acc = %.4f | f1 = %.4f'%
            (self.model.updates, self.n_examples,
             self.model.train_loss, self.model.train_acc, self.model.train_f1))

