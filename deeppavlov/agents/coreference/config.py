# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def add_cmdline_args(parser):
    """
    Add parameters from command line.
    Args:
        parser: parameters parser

    Returns:
        nothing
    """
    # Runtime environment
    agent = parser.add_argument_group('Coreference Arguments')

    # Computation limits.
    agent.add_argument('--mention_ratio', type=float, default=0.4)
    agent.add_argument('--max_antecedents', type=int, default=250)
    agent.add_argument('--max_training_sentences', type=int, default=50)

    # Model hyperparameters.
    agent.add_argument('--filter_widths', type=list, default=[3, 4, 5])
    agent.add_argument('--filter_size', type=int, default=50)
    agent.add_argument('--char_embedding_size', type=int, default=8)
    agent.add_argument('--lstm_size', type=int, default=200)
    agent.add_argument('--ffnn_size', type=int, default=150)
    agent.add_argument('--ffnn_depth', type=int, default=2)
    agent.add_argument('--feature_size', type=int, default=20)
    agent.add_argument('--max_mention_width', type=int, default=10)
    agent.add_argument('--use_features', type=bool, default=True)
    agent.add_argument('--model_heads', type=bool, default=True)
    agent.add_argument('--use_metadata', type=bool, default=True)

    # Learning hyperparameters.
    agent.add_argument('--learning_rate', type=float, default=0.001)
    agent.add_argument('--decay_frequency', type=float, default=100)
    agent.add_argument('--decay_rate', type=float, default=0.999)
    agent.add_argument('--final_rate', type=float, default=0.0002)  # ~150k iteration
    agent.add_argument('--max_gradient_norm', type=float, default=5.0)
    agent.add_argument('--optimizer', type=str, default='adam')
    agent.add_argument('--dropout_rate', type=float, default=0.2)
    agent.add_argument('--lexical_dropout_rate', type=float, default=0.5)

    # Other.
    agent.add_argument('--pretrained_model', type='bool', default=False)
    agent.add_argument('--embedding_size', type=int, default=100)
    agent.add_argument('--emb_format', type=str, default='vec')
    agent.add_argument('--genres', type=list, default=['bc'])
    agent.add_argument('--emb_lowercase', type='bool', default=False)
    agent.add_argument('--name', type=str, default='main')
    agent.add_argument('--rep_iter', type=int, default=144)
    agent.add_argument('--train_on_gold', default=False)
    agent.add_argument('--random_seed', type=int, default=0)
