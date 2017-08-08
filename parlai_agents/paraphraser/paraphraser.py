import copy

from . import config

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
        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        # # Some examples will be None (no answer found). Filter them.
        # examples = [self._build_ex(obs) for obs in observations]
        # valid_inds = [i for i in range(batch_size) if examples[i] is not None]
        # examples = [ex for ex in examples if ex is not None]
        #
        # # If all examples are invalid, return an empty batch.
        # if len(examples) == 0:
        #     return batch_reply
        #
        # # Else, use what we have (hopefully everything).
        # batch = self.batchify(
        #     examples, null=self.word_dict[self.word_dict.null_token], cuda=self.opt['cuda']
        # )
        #
        # # Either train or predict
        # if 'labels' in observations[0]:
        #     self.n_examples += len(examples)
        #     self.model.update(batch)
        # else:
        #     predictions = self.model.predict(batch)
        #     for i in range(len(predictions)):
        #         batch_reply[valid_inds[i]]['text'] = predictions[i]

        return batch_reply
