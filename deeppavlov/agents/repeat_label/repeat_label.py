from parlai.core.agents import Agent
import random


class RepeatLabelAgent(Agent):

    def __init__(self, opt, shared=None):
        self.returnOneRandomAnswer = opt.get('returnOneRandomAnswer', True)
        self.cantAnswerPercent = opt.get('cantAnswerPercent', 0)
        self.id = 'RepeatLabelAgent'
        super().__init__(opt, shared)

    def act(self):
        obs = self.observation
        if obs is None:
            return {'text': "Nothing to repeat yet."}
        reply = {'id': self.getID()}
        if 'labels' in obs and obs['labels'] is not None and len(obs['labels']) > 0:
            labels = obs['labels']
            if random.random() >= self.cantAnswerPercent:
                if self.returnOneRandomAnswer:
                    reply['text'] = labels[random.randrange(len(labels))]
                else:
                    reply['text'] = ', '.join(labels)
            else:
                # Some 'self.cantAnswerPercent' percentage of the time
                # the agent does not answer.
                reply['text'] = "I don't know."
        else:
            reply['text'] = "I don't know."

        if 'label_candidates' in obs and len(obs['label_candidates']) > 0:
            # Produce text_candidates by selecting random candidate labels.
            reply['text_candidates'] = [reply['text']]
            reply['text_candidates'].extend(random.sample(
                obs['label_candidates'], min(len(obs['label_candidates']), 99)))
        return reply
