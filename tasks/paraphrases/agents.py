from parlai.core.dialog_teacher import DialogTeacher
from .build import build
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import KFold
import random


def _path(opt):
    # ensure data is built
    build(opt)

    # set up paths to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    fname = 'paraphrases'
    if dt == 'test':
        fname += '_gold'
    fname += '.xml'
    datafile = os.path.join(opt['datapath'], 'paraphrases', fname)
    return datafile


class DefaultTeacher(DialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        teacher = argparser.add_argument_group('paraphrases teacher arguments')
        teacher.add_argument('--cross-validation-seed', type=int, default=71)
        teacher.add_argument('--cross-validation-model-index', type=int)
        teacher.add_argument('--cross-validation-splits-count', type=int, default=5)

    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype_strict = opt['datatype'].split(':')[0]

        opt['datafile'] = _path(opt)

        # store identifier for the teacher in the dialog
        self.id = 'paraphrases_teacher'

        # define standard question, since it doesn't change for this task
        self.question = "Эти два предложения — парафразы?"
        self.answer_candidates = ['Да', 'Нет']

        random_state = random.getstate()
        random.seed(opt.get('cross_validation_seed'))
        self.random_state = random.getstate()
        random.setstate(random_state)

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
            context = ET.iterparse(labels_file, events=("start", "end"))

            # turn it into an iterator
            context = iter(context)

            # get the root element
            event, root = next(context)

            for event, elem in context:
                if event == "end" and elem.tag == "paraphrase":
                    question = []
                    for child in elem.iter():
                        if child.get('name') == 'text_1':
                            question.append(child.text)
                        if child.get('name') == 'text_2':
                            question.append(child.text)
                        if child.get('name') == 'class':
                            y.append(['Да' if int(child.text) >= 0 else 'Нет'])
                    root.clear()
                    questions.append("\n".join(question))

        episode_done = True
        if not y:
            y = [None for _ in range(len(questions))]

        indexes = range(len(questions))
        if self.datatype_strict != 'test':
            random_state = random.getstate()
            random.setstate(self.random_state)
            kf_seed = random.randrange(500000)
            kf = KFold(self.opt.get('cross_validation_splits_count'), shuffle=True,
                       random_state=kf_seed)
            i = 0
            for train_index, test_index in kf.split(questions):
                indexes = train_index if self.datatype_strict == 'train' else test_index
                if i >= self.opt.get('cross_validation_model_index', 0):
                    break
            self.random_state = random.getstate()
            random.setstate(random_state)

        # define iterator over all queries
        for i in indexes:
            # get current label, both as a digit and as a text
            # yield tuple with information and episode_done? flag
            yield (self.question + "\n" + questions[i], y[i]), episode_done

    def reset(self):
        super().reset()

        random_state = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(self.data.data)
        self.random_state = random.getstate()
        random.setstate(random_state)
