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
    """Add command line arguments for NER model"""
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
