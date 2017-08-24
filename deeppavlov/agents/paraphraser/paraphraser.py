import copy

from parlai.core.agents import Agent

from . import config
from .embeddings_dict import EmbeddingsDict
from .model import ParaphraserModel


def prediction2text(prediction):
    return 'Да' if prediction > 0.5 else 'Нет'


def predictions2text(predictions):
    return [prediction2text(ex) for ex in predictions]


class EnsembleParaphraserAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        ensemble = argparser.add_argument_group('Ensemble parameters')
        ensemble.add_argument('--model_files', type=str, default=None, nargs='+',
                              help='list of all the model files for the ensemble')

    def __init__(self, opt, shared=None):
        self.id = 'ParaphraserAgent'
        self.episode_done = True
        super().__init__(opt, shared)
        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        embdict = EmbeddingsDict(opt, opt.get('embedding_dim'))
        self.models = []
        for model_file in opt.get('model_files', []):
            opt['pretrained_model'] = model_file
            self.models.append(ParaphraserModel(opt, embdict))

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
        predictions = [[] for _ in range(batch_size)]
        for model in self.models:
            examples = [model.build_ex(obs) for obs in observations]
            valid_inds = [i for i in range(batch_size) if examples[i] is not None]
            examples = [ex for ex in examples if ex is not None]
            batch, _ = model.batchify(examples)
            prediction = model.predict(batch)
            for i in range(len(prediction)):
                predictions[valid_inds[i]].append(prediction[i])
        for i in range(batch_size):
            if len(predictions[i]):
                prediction = sum(predictions[i])/len(predictions[i])
                batch_reply[i]['text'] = prediction2text(prediction)
                batch_reply[i]['score'] = prediction

        return batch_reply


class ParaphraserAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        self.id = 'ParaphraserAgent'
        self.episode_done = True
        super().__init__(opt, shared)
        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
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
        examples = [self.model.build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batch_size) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]
        batch = self.model.batchify(examples)

        if 'labels' in observations[0] and not self.opt.get('pretrained_model'):
            self.n_examples += len(examples)
            self.model.update(batch)
        else:
            batch, _ = batch
            predictions = self.model.predict(batch)
            texts = predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = texts[i]
                batch_reply[valid_inds[i]]['score'] = predictions[i]

        return batch_reply

    def save(self, fname=None):
        """Save the parameters of the agent to a file."""
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save(fname)

    def report(self):
        return (
            '[train] updates = %d | exs = %d | loss = %.4f | acc = %.4f | f1 = %.4f'%
            (self.model.updates, self.n_examples,
             self.model.train_loss, self.model.train_acc, self.model.train_f1))

    def reset_metrics(self):
        self.model.reset_metrics()
        self.n_examples = 0

    def shutdown(self):
        if not self.is_shared:
            if self.model is not None:
                self.model.shutdown()
            self.model = None
