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
        reply = self.observation
        reply['id'] = self.id
        return reply
