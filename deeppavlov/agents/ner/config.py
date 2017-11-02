def add_cmdline_args(parser):
    # Runtime environment
    agent = parser.add_argument_group('NER Agent Arguments')
    agent.add_argument('--pretrained-model', type=str)
    agent.add_argument('--cuda', type='bool', default=False)
    agent.add_argument('--gpu', type=int, default=-1)
    agent.add_argument('--random_seed', type=int, default=42)
    agent.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                        help='Learning rate for SGD (default 0.1)')

    agent.add_argument('--trainer_type', type=str, default='naive',
                       help='Training algorithm: beam or naive (default)')
    agent.add_argument('--beam_size', type=int, default=8)
