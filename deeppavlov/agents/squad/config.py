import os
import sys
import logging

def add_cmdline_args(parser):
    # Runtime environment
    agent = parser.add_argument_group('Paraphraser Arguments')
    agent.add_argument('--no_cuda', type='bool', default=False)
    agent.add_argument('--gpu', type=int, default=-1)
    agent.add_argument('--random_seed', type=int, default=1013)

    # Basics
    agent.add_argument('--pretrained_model', type=str, default=None,
                       help='Load dict/features/weights/opts from this file prefix')
    agent.add_argument('--log_file', type=str, default=None)
    agent.add_argument('--model_file', type=str, default=None,
                       help='Save dict/features/weights/opts from this file')
    agent.add_argument('--fasttext_model', type=str, default=None,
                       help='fasttext trained model file name')

    # Model details
    agent.add_argument('--model_name', default='squad default')
    agent.add_argument('--max_context_length', type=int, default=300)
    agent.add_argument('--max_question_length', type=int, default=30)
    agent.add_argument('--embedding_dim', type=int, default=300)
    agent.add_argument('--learning_rate', type=float, default=1e-5)
    agent.add_argument('--epoch_num', type=int, default=1)
    agent.add_argument('--seed', type=int, default=243)
    agent.add_argument('--hidden_dim', type=int, default=200)
    agent.add_argument('--attention_dim', type=int, default=25)
    agent.add_argument('--perspective_num', type=int, default=10)
    agent.add_argument('--aggregation_dim', type=int, default=200)
    agent.add_argument('--dense_dim', type=int, default=50)
    agent.add_argument('--ldrop_val', type=float, default=0.0)
    agent.add_argument('--dropout_val', type=float, default=0.0)
    agent.add_argument('--recdrop_val', type=float, default=0.0)
    agent.add_argument('--inpdrop_val', type=float, default=0.0)

    # Basics
    agent.add_argument('--embedding_file', type=str, default=None,
                        help='File of space separated embeddings: w e1 ... ed')

    # Model-specific
    agent.add_argument('--concat_rnn_layers', type='bool', default=True)
    agent.add_argument('--question_merge', type=str, default='self_attn',
                        help='The way of computing question representation')
    agent.add_argument('--use_qemb', type='bool', default=True,
                        help='Whether to use weighted question embeddings')
    agent.add_argument('--use_in_question', type='bool', default=True,
                        help='Whether to use in_question features')
    agent.add_argument('--use_tf', type='bool', default=True,
                        help='Whether to use tf features')
    agent.add_argument('--use_time', type=int, default=0,
                        help='Time features marking how recent word was said')


def set_defaults(opt):
    # Embeddings options
    if opt.get('embedding_file'):
        if not os.path.isfile(opt['embedding_file']):
            raise IOError('No such file: %s' % opt['embedding_file'])
        with open(opt['embedding_file']) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        opt['embedding_dim'] = dim
    elif not opt.get('embedding_dim'):
        raise RuntimeError(('Either embedding_file or embedding_dim '
                            'needs to be specified.'))

def override_args(opt, override_opt):
    # Major model args are reset to the values in override_opt.
    # Non-architecture args (like dropout) are kept.
    args = set(['embedding_file', 'embedding_dim', 'hidden_size', 'doc_layers',
                'question_layers', 'rnn_type', 'optimizer', 'concat_rnn_layers',
                'question_merge', 'use_qemb', 'use_in_question', 'use_tf',
                'vocab_size', 'num_features', 'use_time'])
    for k, v in override_opt.items():
        if k in args:
            opt[k] = v

