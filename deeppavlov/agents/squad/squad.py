import copy
import os
import pickle
import numpy as np
from . import config
from .model import SquadModel
from parlai.core.agents import Agent
from parlai.core.params import class2str
from .embeddings_dict import SimpleDictionaryAgent
from .utils import build_feature_dict, vectorize, batchify, load_embeddings

class SquadAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        SquadAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return SimpleDictionaryAgent

    def __init__(self, opt, shared=None):
        if opt['numthreads'] >1:
            raise RuntimeError("numthreads > 1 not supported for this model.")

        # Load dict.
        if not shared:
            word_dict = SquadAgent.dictionary_class()(opt)
        # All agents keep track of the episode (for multiple questions)
        self.episode_done = True

        # Only create an empty dummy class when sharing
        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        self.id = self.__class__.__name__
        self.word_dict = word_dict
        self.opt = copy.deepcopy(opt)
        config.set_defaults(self.opt)

        if self.opt.get('model_file') and os.path.isfile(opt['model_file']):
            self._init_from_saved(opt['model_file'])
        else:
            if self.opt.get('pretrained_model'):
                self._init_from_saved(opt['pretrained_model'])
            else:
                self._init_from_scratch()

        self.embeddings = load_embeddings(opt, word_dict)
        self.n_examples = 0


    def _init_from_scratch(self):
        '''
        Initializes model from scratch
        '''
        self.feature_dict = build_feature_dict(self.opt)
        self.opt['num_features'] = len(self.feature_dict)
        self.opt['vocab_size'] = len(self.word_dict)

        print('[ Initializing model from scratch ]')
        self.model = SquadModel(self.opt, self.word_dict, self.feature_dict)


    def _init_from_saved(self, fname):
        '''
        Loading model from checkpoint.
        '''
        print('[ Loading from saved %s ]' % fname)

        with open(fname+'.pkl','rb') as f:
            saved_params = pickle.load(f)

        # TODO expand dict and embeddings for new data
        self.word_dict = saved_params['word_dict']
        self.feature_dict = saved_params['feature_dict']
        config.override_args(self.opt, saved_params['config'])

        self.model = SquadModel(self.opt, self.word_dict, self.feature_dict, fname)


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
        """Update or predict on a single example (batchsize = 1)."""
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        reply = {'id': self.getID()}

        ex = self._build_ex(self.observation)
        if ex is None:
            return reply
        batch = batchify(
            [ex], null=self.word_dict[self.word_dict.null_token]
        )

        # Either train or predict
        if 'labels' in self.observation and not self.opt.get('pretrained_model'):
            self.n_examples += 1
            self.model.update(batch)
        else:
            reply['text'] = self.model.predict(batch)[0]

        return reply

    def batch_act(self, observations):

        """Update or predict on a batch of examples.
        More efficient than act().
        """
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # Some examples will be None (no answer found). Filter them.
        examples = [self._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batchsize) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]

        # If all examples are invalid, return an empty batch.
        if len(examples) == 0:
            return batch_reply

        # Else, use what we have (hopefully everything).
        batch = batchify(
            examples, null=self.word_dict[self.word_dict.null_token]
        )

        # Either train or predict
        if 'labels' in observations[0]:
            self.n_examples += len(examples)
            self.model.update(batch)
        else:
            predictions = self.model.predict(batch)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]

        return batch_reply

    def drop_lr(self):
        ''' Reset optimizer and reset learning rate if validation score is not increasing'''
        self.model.model.optimizer.lr = self.model.model.optimizer.lr * self.opt['lr_drop']

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save(fname)

    def report(self):

        output = (
            '[train] updates = %d | exs = %d | loss = %.4f | acc = %.4f | f1 = %.4f | em = %.4f'%
            (self.model.updates, self.n_examples,
             self.model.train_loss.avg, self.model.train_acc.avg, self.model.train_f1.avg, self.model.train_em.avg))

        self.model.train_loss.reset()
        self.model.train_acc.reset()
        self.model.train_f1.reset()
        self.model.train_em.reset()

        return output


    # --------------------------------------------------------------------------
    # Helper functions.
    # --------------------------------------------------------------------------

    def _build_ex(self, ex):
        """Find the token span of the answer in the context for this example.
        If a token span cannot be found, return None. Otherwise, torchify.
        """
        # Check if empty input (end of epoch)
        if not 'text' in ex:
            return

        # Split out document + question
        inputs = {}
        fields = ex['text'].strip().split('\n')

        # Data is expected to be text + '\n' + question
        if len(fields) < 2:
            raise RuntimeError('Invalid input. Is task a QA task?')

        document, question = ' '.join(fields[:-1]), fields[-1]
        inputs['document'] = self.word_dict.tokenize(document)
        inputs['question'] = self.word_dict.tokenize(question)
        inputs['target'] = None

        # Find targets (if labels provided).
        # Return if we were unable to find an answer.
        if 'labels' in ex:
            inputs['target'] = self._find_target(inputs['document'],
                                                 ex['labels'])
            if inputs['target'] is None:
                return

        # Vectorize.
        inputs = vectorize(self.opt, inputs, self.word_dict, self.feature_dict, self.embeddings)

        # Return inputs with original text + spans (keep for prediction)
        return inputs + (document, self.word_dict.span_tokenize(document))

    def _find_target(self, document, labels):
        """Find the start/end token span for all labels in document.
        Return a random one for training.
        """
        def _positions(d, l):
            for i in range(len(d)):
                for j in range(i, min(len(d) - 1, i + len(l))):
                    if l == d[i:j + 1]:
                        yield(i, j)
        targets = []
        for label in labels:
            targets.extend(_positions(document, self.word_dict.tokenize(label)))
        if len(targets) == 0:
            return
        return targets[np.random.choice(len(targets))]
