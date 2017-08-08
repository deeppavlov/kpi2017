from parlai.core.dialog_teacher import DialogTeacher
from .build import build
import os
import xml.etree.ElementTree as ET


def _path(opt):
    # ensure data is built
    build(opt)

    # set up paths to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    fname = 'paraphrases'
    if dt == 'valid':
        fname += '_gold'
    elif dt == 'test':
        fname += '_test'
    fname += '.xml'
    datafile = os.path.join(opt['datapath'], 'paraphrases', fname)
    return datafile


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype = opt['datatype'].split(':')[0]

        opt['datafile'] = _path(opt)

        # store identifier for the teacher in the dialog
        self.id = 'paraphrases_teacher'

        # define standard question, since it doesn't change for this task
        self.question = "Эти два предложения — парафразы?"
        self.answer_candidates = ['Да', 'Нет']

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

        # define iterator over all queries
        for i in range(len(questions)):
            # get current label, both as a digit and as a text
            # yield tuple with information and episode_done? flag
            yield (self.question + "\n" + questions[i], y[i]), episode_done
