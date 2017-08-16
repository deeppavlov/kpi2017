def add_cmdline_args(parser):
    # Runtime environment
    agent = parser.add_argument_group('DrQA Arguments')
    agent.add_argument('--cuda', type='bool', default=False)
    agent.add_argument('--gpu', type=int, default=-1)
    # agent.add_argument('--random_seed', type=int, default=1013)
    #
    # # Basics
    # agent.add_argument('--embedding_file', type=str, default=None,
    #                     help='File of space separated embeddings: w e1 ... ed')
    # agent.add_argument('--pretrained_model', type=str, default=None,
    #                     help='Load dict/features/weights/opts from this file')
    # agent.add_argument('--log_file', type=str, default=None)
    #
    # # Model details
    # agent.add_argument('--fix_embeddings', type='bool', default=True)
    # agent.add_argument('--tune_partial', type=int, default=0,
    #                     help='Train the K most frequent word embeddings')
    # agent.add_argument('--embedding_dim', type=int, default=300,
    #                     help=('Default embedding size if '
    #                           'embedding_file is not given'))
    # agent.add_argument('--hidden_size', type=int, default=128,
    #                     help='Hidden size of RNN units')
    # agent.add_argument('--doc_layers', type=int, default=3,
    #                     help='Number of RNN layers for passage')
    # agent.add_argument('--question_layers', type=int, default=3,
    #                     help='Number of RNN layers for question')
    # agent.add_argument('--rnn_type', type=str, default='lstm',
    #                     help='RNN type: lstm (default), gru, or rnn')
    #
    # Optimization details
    agent.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                        help='Learning rate for SGD (default 0.1)')
    # agent.add_argument('--weight_decay', type=float, default=0,
    #                     help='Weight decay (default 0)')
    # agent.add_argument('--momentum', type=float, default=0,
    #                     help='Momentum (default 0)')
    # agent.add_argument('--grad_clipping', type=float, default=10,
    #                     help='Gradient clipping (default 10.0)')
    # agent.add_argument('--valid_metric', type=str,
    #                     choices=['accuracy', 'f1'], default='f1',
    #                     help='Metric for choosing best valid model')
    # agent.add_argument('--max_len', type=int, default=15,
    #                     help='The max span allowed during decoding')
    # agent.add_argument('--rnn_padding', type='bool', default=False)
    # agent.add_argument('--display_iter', type=int, default=10,
    #                     help='Print train error after every \
    #                           <display_iter> epoches (default 10)')
    # agent.add_argument('--dropout_emb', type=float, default=0.4,
    #                     help='Dropout rate for word embeddings')
    # agent.add_argument('--dropout_rnn', type=float, default=0.4,
    #                     help='Dropout rate for RNN states')
    # agent.add_argument('--dropout_rnn_output', type='bool', default=True,
    #                     help='Whether to dropout the RNN output')
    # agent.add_argument('--optimizer', type=str, default='adamax',
    #                     help='Optimizer: sgd or adamax (default)')
    #
    # Model-specific
    agent.add_argument('--trainer_type', type=str, default='naive',
                       help='Training algorithm: beam or naive (default)')
    agent.add_argument('--beam_size', type=int, default=8)
    # agent.add_argument('--concat_rnn_layers', type='bool', default=True)
    # agent.add_argument('--question_merge', type=str, default='self_attn',
    #                     help='The way of computing question representation')
    # agent.add_argument('--use_qemb', type='bool', default=True,
    #                     help='Whether to use weighted question embeddings')
    # agent.add_argument('--use_in_question', type='bool', default=True,
    #                     help='Whether to use in_question features')
    # agent.add_argument('--use_tf', type='bool', default=True,
    #                     help='Whether to use tf features')
    # agent.add_argument('--use_time', type=int, default=0,
    #                     help='Time features marking how recent word was said')
