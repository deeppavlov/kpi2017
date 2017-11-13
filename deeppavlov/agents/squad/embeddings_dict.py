"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

try:
    import spacy
except ImportError:
    raise ImportError(
        "Please install spacy and spacy 'en' model: go to spacy.io"
    )

from parlai.core.dict import DictionaryAgent
from .utils import normalize_text
import urllib


NLP = spacy.load('en')


class SimpleDictionaryAgent(DictionaryAgent):
    """Overrides DictionaryAgent to use spaCy tokenizer."""

    @staticmethod
    def add_cmdline_args(argparser):
        """Specify permission to index words not included in embedding_file."""
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--pretrained_words', type='bool', default=True,
            help='Use only words found in provided embedding_file'
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Index words in embedding file
        if self.opt['pretrained_words'] and self.opt.get('embedding_file'):
            if not os.path.exists(self.opt.get('embedding_file')):
                emb_url = os.environ.get('EMBEDDINGS_URL')
                if not emb_url:
                    raise RuntimeError('No glove embeddings provided')
                fname = os.path.basename(self.opt.get('embedding_file'))
                try:
                    print('Trying to download a glove embeddings from the server')
                    urllib.request.urlretrieve(urllib.parse.urljoin(emb_url, fname), self.opt.get('embedding_file'))
                    print('Downloaded a glove embeddings')
                except Exception as e:
                    raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable is set incorrectly', e)

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
        """
        Args:
            text: string to tokenize
            **kwargs: anything

        Returns:
            list of tokens (words, punctuation, etc...)
        """
        tokens = NLP.tokenizer(text)
        return [t.text for t in tokens]

    def span_tokenize(self, text):
        """
        Args:
            text: string to tokenize

        Returns:
            list of tuples with start and end position of each token in original string
        """
        tokens = NLP.tokenizer(text)
        return [(t.idx, t.idx + len(t.text)) for t in tokens]

    def add_to_dict(self, tokens):
        """Builds dictionary from the list of provided tokens.
        Only adds words contained in self.embedding_words, if not None."""

        for token in tokens:
            if (self.embedding_words is not None and
                token not in self.embedding_words):
                continue
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token


