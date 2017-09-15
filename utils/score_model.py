# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train a model.

After training, computes validation and test error.

Run with, e.g.:

python examples/train_model.py -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model

..or..

python examples/train_model.py -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128

..or..

python examples/train_model.py -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

TODO List:
- More logging (e.g. to files), make things prettier.
"""
import copy
import math
import pickle
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
from parlai.core.worlds import create_task

from utils import build_dict

def run_eval(agent, opt, datatype, max_exs=-1, write_log=False, valid_world=None):
    """Eval on validation/test data.
    - Agent is the agent to use for the evaluation.
    - opt is the options that specific the task, eval_task, etc
    - datatype is the datatype to use, such as "valid" or "test"
    - write_log specifies to write metrics to file if the model_file is set
    - max_exs limits the number of examples if max_exs > 0
    - valid_world can be an existing world which will be reset instead of reinitialized
    """
    print('[ running eval: ' + datatype + ' ]')
    opt['datatype'] = datatype
    if opt.get('evaltask'):

        opt['task'] = opt['evaltask']

    if valid_world is None:
        valid_world = create_task(opt, agent)
    else:
        valid_world.reset()
    cnt = 0
    for _ in valid_world:
        valid_world.parley()
        if cnt == 0 and opt['display_examples']:
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        cnt += opt['batchsize']
        if valid_world.epoch_done() or (max_exs > 0 and cnt > max_exs):
            # note this max_exs is approximate--some batches won't always be
            # full depending on the structure of the data
            break
    valid_report = valid_world.report()

    metrics = datatype + ':' + str(valid_report)
    print(metrics)
    if write_log and opt['model_file']:
        # Write out metrics
        f = open(opt['model_file'] + '.' + datatype, 'w')
        f.write(metrics + '\n')
        f.close()

    return valid_report, valid_world


def train_model(opt):
    # Possibly build a dictionary (not all models do this).
    if opt['dict_build_first'] and 'dict_file' in opt:
        if opt['dict_file'] is None and opt.get('pretrained_model'):
            opt['dict_file'] = opt['pretrained_model'] + '.dict'
        if opt['dict_file'] is None and opt.get('model_file'):
            opt['dict_file'] = opt['model_file'] + '.dict'
        print("[ building dictionary first... ]")
        build_dict.build_dict(opt)

    # Create model and assign it to the specified task
    agent = create_agent(opt)

    print('[ testing... ]')

    with open(opt['pretrained_model']+'.pkl', 'rb') as f:
        file = pickle.load(f)

    agent = create_agent(opt)


    run_eval(agent, opt, 'test', write_log=True)

    agent.shutdown()




def main(args=None):
    # Get command line arguments
    parser = ParlaiParser(True, True)
    train = parser.add_argument_group('Training Loop Arguments')
    train.add_argument('-et', '--evaltask',
                        help=('task to use for valid/test (defaults to the ' +
                              'one used for training if not set)'))
    train.add_argument('-d', '--display-examples',
                        type='bool', default=False)
    train.add_argument('-e', '--num-epochs', type=float, default=-1)
    train.add_argument('-ttim', '--max-train-time',
                        type=float, default=-1)
    train.add_argument('-ltim', '--log-every-n-secs',
                        type=float, default=2)
    train.add_argument('-le', '--log-every-n-epochs',
                        type=int, default=0)
    train.add_argument('-vtim', '--validation-every-n-secs',
                        type=float, default=-1)
    train.add_argument('-ve', '--validation-every-n-epochs',
                        type=int, default=0)
    train.add_argument('-vme', '--validation-max-exs',
                        type=int, default=-1,
                        help='max examples to use during validation (default ' +
                             '-1 uses all)')
    train.add_argument('-vp', '--validation-patience',
                        type=int, default=5,
                        help=('number of iterations of validation where result '
                              + 'does not improve before we stop training'))
    train.add_argument('-dbf', '--dict-build-first',
                        type='bool', default=True,
                        help='build dictionary first before training agent')
    train.add_argument('--chosen-metric', default='accuracy',
                       help='metric with which to measure improvement')
    train.add_argument('--lr-drop', '--lr-drop-patience', type=int, default=-1,
                       help='drop learning rate if validation metric is not improving')
    opt = parser.parse_args(args=args)
    if opt.get('cross_validation_splits_count', 0) > 1 and opt.get('cross_validation_model_index') is None:
        train_cross_valid(opt)
    else:
        train_model(opt)


if __name__ == '__main__':
    main()
