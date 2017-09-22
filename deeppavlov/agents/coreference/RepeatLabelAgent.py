from parlai.core.agents import Agent


class RepeatLabelAgent(Agent):
    # #
    # initialize by setting id
    # #
    def __init__(self, opt):
        self.id = 'LabelAgent'
    # #
    # store observation for later, return it unmodified
    # #
    def observe(self, observation):
        self.observation = observation
        return observation
    # #
    # return label from before if available
    # #
    def act(self):
        reply = {'id': self.id}
        if 'labels' in self.observation:
            reply['text'] = ', '.join(self.observation['labels'])
        else:
            reply['text'] = "I don't know."
        return reply
