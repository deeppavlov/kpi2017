import os
import copy
import numpy as np

try:
    import spacy
except ImportError:
    raise ImportError(
        "Please install spacy and spacy 'en' model: go to spacy.io"
    )


from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from . import config
from .utils import build_feature_dict, vectorize, batchify, normalize_text


NLP = spacy.load('en')


class SimpleDictionaryAgent(DictionaryAgent):
    """Override DictionaryAgent to use spaCy tokenizer."""

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--pretrained_words', type='bool', default=True,
            help='Use only words found in provided embedding_file'
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Index words in embedding file
        if self.opt['pretrained_words'] and self.opt.get('embedding_file'):
            print('[ Indexing words with embeddings... ]')
            self.embedding_words = set()
            with open(self.opt['embedding_file']) as f:
                for line in f:
                    w = normalize_text(line.rstrip().split(' ')[0])
                    self.embedding_words.add(w)
            print('[ Num words in set = %d ]' %
                  len(self.embedding_words))
        else:
            self.embedding_words = None

    def tokenize(self, text, **kwargs):
        tokens = NLP.tokenizer(text)
        return [t.text for t in tokens]

    def span_tokenize(self, text):
        tokens = NLP.tokenizer(text)
        return [(t.idx, t.idx + len(t.text)) for t in tokens]

    def add_to_dict(self, tokens):
        """Builds dictionary from the list of provided tokens.
        Only adds words contained in self.embedding_words, if not None.
        """
        for token in tokens:
            if (self.embedding_words is not None and
                token not in self.embedding_words):
                continue
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token


