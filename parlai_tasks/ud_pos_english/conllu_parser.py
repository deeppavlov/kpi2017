import re
import sys
from collections import namedtuple


headers = 'id form lemma cpostag postag feats head deprel deps misc'.split()
WordConNLL = namedtuple('WordConNLL', headers)


def read_conllu_file(filename):
    errors_count = 0
    with open(filename) as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if sentence:
                    yield sentence
                    sentence = []
                continue
            try:
                w = WordConNLL(*re.match(r'(\d+.?\d*)' + r'\s+([^\s]+)' * 9, line).groups())
                sentence.append(w)
            except Exception as e:
                print("'{}' happened for '{}'".format(e, line), file=sys.stderr)
                errors_count += 1
                if errors_count > 10:
                    raise
    if sentence:
        yield sentence


def get_pos_tags(sentences):
    for s in sentences:
        new_sent = []
        for w in s:
            p = (w.form, w.cpostag)
            new_sent.append(p)
        yield new_sent


def get_words(sentences):
    for s in sentences:
        yield [w.form for w in s]
