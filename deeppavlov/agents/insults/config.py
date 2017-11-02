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


def add_cmdline_args(parser):
    """Add parameters from command line.

    Args:
        parser: parameters parser

    Returns:
        nothing
    """
    # Runtime environment
    agent = parser.add_argument_group('Insults Arguments')
    agent.add_argument('--no_cuda', type='bool', default=False)
    agent.add_argument('--gpu', type=int, default=-1)
    agent.add_argument('--random_seed', type=int, default=1013)
    agent.add_argument('--model_seed', type=int, default=23)

    # Basics
    agent.add_argument('--embedding_file', type=str, default=None,
                       help='File of space separated embeddings: w e1 ... ed')
    agent.add_argument('--pretrained_model', type=str, default=None,
                       help='Load dict/features/weights/opts from this file')
    agent.add_argument('--log_file', type=str, default=None)
    agent.add_argument('--model_file', type=str, default=None)
    agent.add_argument('--max_sequence_length', type=int, default=100)
    agent.add_argument('--embedding_dim', type=int, default=100)
    agent.add_argument('--learning_rate', type=float, default=1e-1)
    agent.add_argument('--learning_decay', type=float, default=0.)
    agent.add_argument('--seed', type=int, default=243)
    agent.add_argument('--filters_cnn', type=int, default=128)
    agent.add_argument('--units_lstm', type=int, default=128)
    agent.add_argument('--kernel_sizes_cnn',  default='3 3 3')
    agent.add_argument('--regul_coef_conv', type=float, default=1e-3)
    agent.add_argument('--regul_coef_lstm', type=float, default=1e-3)
    agent.add_argument('--regul_coef_dense', type=float, default=1e-3)
    agent.add_argument('--pool_sizes_cnn', default='2 2 2')
    agent.add_argument('--dropout_rate', type=float, default=0.)
    agent.add_argument('--dense_dim', type=int, default=100)

    agent.add_argument('-bd', '--balance_dataset',
                        type='bool', default=False,
                        help='balance train dataset for insult/non-insult?')

    agent.add_argument('-bdr', '--balance_dataset_ratio',
                       type=int, default=1,
                       help='ratio for balancing train dataset for insult/non-insult')

    agent.add_argument('-mn', '--model_name',
                       type=str, default='cnn_word',
                       help='str of models to use. available: cnn_word log_reg svc')

    agent.add_argument('--fasttext_model', type=str, default=None,
                       help='fasttext trained model file name')
    agent.add_argument('-fed', '--fasttext_embeddings_dict', type=str, default=None,
                       help='saved fasttext embeddings dict')
