from parlai.core.dialog_teacher import DialogTeacher
from sklearn.model_selection import KFold

from .build import build
import os
import csv
import sklearn.metrics
import random


def _path(opt):
    # ensure data is built
    build(opt)
    # set up paths to data (specific to each dataset)
    dt = 'test' if opt['datatype'] == 'test' else 'train'
    datafile = os.path.join(opt['datapath'], 'insults', dt + '.csv')
    return datafile


class DefaultTeacher(DialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        teacher = argparser.add_argument_group('Insults teacher arguments')
        teacher.add_argument('--raw-dataset-path', type=str, default=None,
                             help='Path to unprocessed dataset files from Kaggle')
        teacher.add_argument('--teacher-random-seed', type=int, default=270)
        teacher.add_argument('--bagging-fold-index', type=int)
        teacher.add_argument('--bagging-folds-number', type=int, default=5)

    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype_strict = opt['datatype'].split(':')[0]

        opt['datafile'] = _path(opt)

        # store identifier for the teacher in the dialog
        self.id = 'insults_teacher'

        self.answer_candidates = ['Non-insult', "Insult"]

        random_state = random.getstate()
        random.seed(opt.get('teacher_random_seed'))
        self.random_state = random.getstate()
        random.setstate(random_state)

        super().__init__(opt, shared)

        if shared:
            self.observations = shared['observations']
            self.labels = shared['labels']
        else:
            self.observations = []
            self.labels = []

    def share(self):
        shared = super().share()
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

        indexes = range(len(questions))
        if self.datatype_strict != 'test':
            random_state = random.getstate()
            random.setstate(self.random_state)
            kf_seed = random.randrange(500000)
            kf = KFold(self.opt.get('bagging_folds_number'), shuffle=True,
                       random_state=kf_seed)
            i = 0
            for train_index, test_index in kf.split(questions):
                indexes = train_index if self.datatype_strict == 'train' else test_index
                if i >= self.opt.get('bagging_fold_index', 0):
                    break
            self.random_state = random.getstate()
            random.setstate(random_state)

        # define iterator over all queries
        for i in indexes:
            # get current label, both as a digit and as a text
            # yield tuple with information and episode_done? flag
            yield (questions[i], y[i]), episode_done

    def _predictions2text(self, predictions):
        y = ['Insult' if ex > 0.5 else 'Non-insult' for ex in predictions]
        return y

    def _text2predictions(self, predictions):
        y = [1 if ex == 'Insult' else 0 for ex in predictions]
        return y

    def observe(self, observation):
        """Process observation for metrics. """
        if self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            if 'text' in observation.keys():
                self.labels += self._text2predictions(self.lastY)
                self.observations += [observation['score']]
            self.lastY = None
        return observation

    def reset_metrics(self):
        super().reset_metrics()
        del self.observations[:]
        del self.labels[:]

    def report(self):
        loss = sklearn.metrics.log_loss(self.labels, self.observations)
        acc = sklearn.metrics.accuracy_score(self.labels,
                                             self._text2predictions(self._predictions2text(self.observations)))
        try:
            auc = sklearn.metrics.roc_auc_score(self.labels, self.observations)
        except ValueError:
            auc = 0
        report = dict()
        report['comments'] = len(self.observations)
        report['loss'] = loss
        report['accuracy'] = acc
        report['auc'] = auc
        return report

    def reset(self):
        super().reset()

        random_state = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(self.data.data)
        self.random_state = random.getstate()
        random.setstate(random_state)


class FullTeacher(DefaultTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        teacher = argparser.add_argument_group('Insults teacher arguments')
        teacher.add_argument('--raw-dataset-path', type=str, default=None,
                             help='Path to unprocessed dataset files from Kaggle')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

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

        indexes = range(len(questions))

        # define iterator over all queries
        for i in indexes:
            # get current label, both as a digit and as a text
            # yield tuple with information and episode_done? flag
            yield (questions[i], y[i]), episode_done

