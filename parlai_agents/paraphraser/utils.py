import numpy as np


def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors."""
    embeddings_index = {}

    # Fill in embeddings
    if not opt.get('embedding_file'):
        raise RuntimeError('Tried to load embeddings with no embedding file.')
    with open(opt['embedding_file']) as f:
        for line in f:
            values = line.rsplit(sep=' ', maxsplit=opt['embedding_dim'] + 1)
            assert(len(values) == opt['embedding_dim'] + 1)
            word = values[0]
            coefs = np.asarray(values[1:-1], dtype='float32')
            embeddings_index[word] = coefs

    # prepare embedding matrix
    embedding_matrix = np.zeros((len(word_dict) + 1, opt['embedding_dim']))
    for word, i in word_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix