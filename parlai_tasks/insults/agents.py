from parlai.core.dialog_teacher import DialogTeacher
from .build import build
import os
import csv
from keras import backend as K
import numpy as np
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from .metrics import roc_auc_score

def _path(opt):
    # ensure data is built
    build(opt)
    # set up paths to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    datafile = os.path.join(opt['datapath'], 'insults', dt + '.csv')
    return datafile


class DefaultTeacher(DialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Teacher arguments')
        agent.add_argument('--raw-dataset-path', type=str, default=None,
                           help='Path to unprocessed dataset files from Kaggle')

    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype = opt['datatype'].split(':')[0]

        opt['datafile'] = _path(opt)

        # store identifier for the teacher in the dialog
        self.id = 'insults_teacher'

        self.answer_candidates = ['Non-insult', "Insult"]

        super().__init__(opt, shared)

        if shared:
            self.observations = shared['observations']
            self.labels = shared['labels']
        else:
            self.observations = []
            self.labels = []

    def share(self):
        shared = super().share()
        shared['data'] = self.data.share()
        shared['observations'] = self.observations
        shared['labels'] = self.labels
        return shared

    def label_candidates(self):
        return self.answer_candidates

    def setup_data(self, path):
        print('loading: ' + path)

        questions = []
        y = []

        # open data file with labels
        # (path will be provided to setup_data from opt['datafile'] defined above)
        with open(path) as labels_file:
            context = csv.reader(labels_file)
            next(context)

            for item in context:
                label, text = item
                questions.append(text)
                y.append([self.answer_candidates[int(label)]])

        episode_done = True

        # define iterator over all queries
        for i in range(len(questions)):
            # get current label, both as a digit and as a text
            # yield tuple with information and episode_done? flag
            yield (questions[i], y[i]), episode_done

    def _predictions2text(self, predictions):
        y = ['Insult' if ex > 0.5 else 'Non-insult' for ex in predictions]
        return y

    def _text2predictions(self, predictions):
        y = [1. if ex == 'Insult' else 0 for ex in predictions]
        return y

    def observe(self, observation):
        """Process observation for metrics. """
        if self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            if 'text' in observation.keys():
                self.labels += self._text2predictions(self.lastY)
                self.observations += self._text2predictions([observation['text']])
            self.lastY = None
        return observation

    def report(self):

        y = np.array(self.labels).reshape(-1)
        y_pred = np.array(self.observations).reshape(-1)
        y_pred_tensor = K.constant(y_pred, dtype='float64')
        loss = K.eval(binary_crossentropy(y.astype('float'), y_pred_tensor))
        acc = K.eval(binary_accuracy(y.astype('float'), y_pred_tensor))
        auc = roc_auc_score(y, y_pred)
        report = dict()
        report['comments'] = len(self.observations)
        report['loss'] = loss
        report['accuracy'] = acc
        report['auc'] = auc
        #info = ''
        #args = ()
        #info += '\n[model] comments = %d | loss = %.4f | acc = %.4f | auc = %.4f'
        #args += (len(self.observations), loss, acc, auc,)
        return report

