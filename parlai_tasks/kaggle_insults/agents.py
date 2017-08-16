from parlai.core.dialog_teacher import DialogTeacher
from .build import build
import os
import csv


def _path(opt):
    # ensure data is built
    build(opt)

    # set up paths to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    datafile = os.path.join(opt['datapath'], 'kaggle_insults', dt + '.csv')
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
        self.id = 'kaggle_insults_teacher'

        self.answer_candidates = ['Ok', "Not Ok"]

        super().__init__(opt, shared)

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
