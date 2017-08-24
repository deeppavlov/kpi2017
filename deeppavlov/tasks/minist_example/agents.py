from parlai.core.dialog_teacher import DialogTeacher
from parlai.core.agents import Agent
from .build import build
import os
import json
import random

def _path(opt):
    # ensure data is built
    build(opt)

    # set up paths to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    labels_path = os.path.join(opt['datapath'], 'mnist', dt, 'labels.json')
    image_path = os.path.join(opt['datapath'], 'mnist', dt)
    return labels_path, image_path

class MnistQATeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype = opt['datatype'].split(':')[0]

        # _path method explained below, returns paths to images and labels
        labels_path, self.image_path = _path(opt)

        # store path to label data in options dictionary
        opt['datafile'] = labels_path

        # store identifier for the teacher in the dialog
        self.id = 'mnist_qa'

        # strings for the labels in the class (digits)
        # (information specific to this task)
        self.num_strs = ['zero', 'one', 'two', 'three', 'four', 'five',
                'six', 'seven', 'eight', 'nine']

        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)

        # open data file with labels
        # (path will be provided to setup_data from opt['datafile'] defined above)
        with open(path) as labels_file:
            self.labels = json.load(labels_file)

        # define standard question, since it doesn't change for this task
        self.question = 'Which number is in the image?'
        # every episode consists of only one query in this task
        episode_done = True

        # define iterator over all queries
        for i in range(len(self.labels)):
            # set up path to curent image
            img_path = os.path.join(self.image_path, '%05d.bmp' % i)
            # get current label, both as a digit and as a text
            label = [self.labels[i], self.num_strs[int(self.labels[i])]]
            # yield tuple with information and episode_done? flag
            yield (self.question, label, None, None, img_path), episode_done

    def label_candidates(self):
        return [str(x) for x in range(10)] + self.num_strs

class RepeatLabelAgent(Agent):

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.returnOneRandomAnswer = opt.get('returnOneRandomAnswer', True)
        self.cantAnswerPercent = opt.get('cantAnswerPercent', 0)
        self.id = 'RepeatLabelAgent'


    def act(self):
        obs = self.observation
        if obs is None:
            return { 'text': "Nothing to repeat yet." }
        reply = {}
        reply['id'] = self.getID()
        if ('labels' in obs and obs['labels'] is not None and len(obs['labels']) > 0):
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
            reply['text_candidates'] = [ reply['text'] ]
            reply['text_candidates'].extend(random.sample(
                obs['label_candidates'], min(len(obs['label_candidates']), 99)))
        return reply
