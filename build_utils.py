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
import sys
import os

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser, str2class
from parlai.core.utils import Timer
from parlai.core.worlds import DialogPartnerWorld, create_task
from parlai.core.dict import DictionaryAgent


def arg_parse(args=None):
    # Get command line arguments
    parser = ParlaiParser(True, True, model_argv=args)
    train = parser.add_argument_group('Training Loop Arguments')
    train.add_argument('-et', '--evaltask',
                        help=('task to use for valid/test (defaults to the ' +
                              'one used for training if not set)'))
    train.add_argument('-d', '--display-examples',
                        type='bool', default=False)
    train.add_argument('-e', '--num-epochs', type=float, default=-1)
    train.add_argument('-ttim', '--max-train-time',
                        type=float, default=86400)
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
    train.add_argument('--chosen-metrics', default='accuracy',
                       help='metrics chosen to measure improvement') # custom arg
    train.add_argument('--lr-drop', '--lr-drop-patience', type=float, default=-1,
                       help='drop learning rate if validation metric is not improving') # custom arg

    opt = parser.parse_args(args=args)

    return opt


def __build_bag_of_words(opt):
    """Build a dictionary for some models.
    opt is a dictionary returned by arg_parse
    """
    if not opt['dict_build_first'] or not 'dict_file' in opt:
        return

    if opt['dict_file'] is None and opt.get('pretrained_model'):
        opt['dict_file'] = opt['pretrained_model'] + '.dict'
    if opt['dict_file'] is None and opt.get('model_file'):
        opt['dict_file'] = opt['model_file'] + '.dict'
    print("[ building dictionary first... ]")

    if not opt.get('dict_file'):
        print('Tried to build dictionary but `--dict-file` is not set. Set ' +
              'this param so the dictionary can be saved.')
        return
    print('[ setting up dictionary. ]')
    if os.path.isfile(opt['dict_file']):
        # Dictionary already built
        print("[ dictionary already built .]")
        return
    if opt.get('dict_class'):
        # Custom dictionary class
        dictionary = str2class(opt['dict_class'])(opt)
    else:
        # Default dictionary class
        dictionary = DictionaryAgent(opt)
    ordered_opt = copy.deepcopy(opt)
    cnt = 0
    # we use train set to build dictionary
    ordered_opt['datatype'] = 'train:ordered'
    if 'stream' in opt['datatype']:
        ordered_opt['datatype'] += ':stream'
    ordered_opt['numthreads'] = 1
    ordered_opt['batchsize'] = 1
    world_dict = create_task(ordered_opt, dictionary)
    # pass examples to dictionary
    for _ in world_dict:
        cnt += 1
        if cnt > opt['dict_maxexs'] and opt['dict_maxexs'] > 0:
            print('Processed {} exs, moving on.'.format(opt['dict_maxexs']))
            # don't wait too long...
            break
        world_dict.parley()
    print('[ dictionary built. ]')
    dictionary.save(opt['dict_file'], sort=True)
    # print('[ num words =  %d ]' % len(dictionary))


def __evaluate_model(valid_world, batchsize, datatype, display_examples, max_exs=-1):
    """Evaluate on validation/test data.
    - valid_world created before calling this function
    - batchsize obtained from opt['batchsize']
    - datatype is the datatype to use, such as "valid" or "test"
    - display_examples is bool
    - max_exs limits the number of examples if max_exs > 0
    """
    print('[ running eval: ' + datatype + ' ]')

    valid_world.reset()
    cnt = 0
    for _ in valid_world:
        valid_world.parley()
        if cnt == 0 and display_examples:
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        cnt += batchsize
        if valid_world.epoch_done() or (max_exs > 0 and cnt >= max_exs):
            # note this max_exs is approximate--some batches won't always be
            # full depending on the structure of the data
            break
    valid_report = valid_world.report()

    print(datatype + ':' + str(valid_report))

    return valid_report, valid_world


def __train_log(opt, world, agent, input_train_dict):
    """Log training procedure.
    - opt is a dictionary returned by arg_parse
    - world used for training
    - agent to be trained
    - train_dict is dictionary of parameters for training, logging, intermediate validation
    """
    train_dict = copy.deepcopy(input_train_dict)

    if (opt['log_every_n_secs'] <= 0 or opt['log_every_n_secs'] >= train_dict['log_time'].time()) and \
            (opt['log_every_n_epochs'] <= 0 or not train_dict['new_epoch'] or
                     (train_dict['epochs_done'] % opt['log_every_n_epochs']) != 0):
            return world, agent, train_dict

    if opt['display_examples']:
        print(world.display() + '\n~~')

    logs = list()
    # time elapsed
    logs.append('time:{}s'.format(math.floor(train_dict['train_time'].time())))
    logs.append('parleys:{}'.format(train_dict['parleys']))
    if train_dict['epochs_done'] > 0:
        logs.append('epochs done:{}'.format(train_dict['epochs_done']))

    # get report and update total examples seen so far
    if hasattr(agent, 'report'):
        train_dict['train_report_agent'] = agent.report()
        train_dict['train_report'] = train_dict['train_report_agent']
        agent.reset_metrics()
    else:
        train_dict['train_report_world'] = world.report()
        train_dict['train_report'] = train_dict['train_report_world']
        world.reset_metrics()
    if hasattr(train_dict['train_report'], 'get') and train_dict['train_report'].get('total'):
        train_dict['total_exs'] += train_dict['train_report']['total']
        logs.append('total_exs:{}'.format(train_dict['total_exs']))

    # check if we should log amount of time remaining
    time_left = None
    if opt['num_epochs'] > 0 and train_dict['total_exs'] > 0:
        exs_per_sec = train_dict['train_time'].time() / train_dict['total_exs']
        time_left = (train_dict['max_exs'] - train_dict['total_exs']) * exs_per_sec
    if opt['max_train_time'] > 0:
        other_time_left = opt['max_train_time'] - train_dict['train_time'].time()
        if time_left is not None:
            time_left = min(time_left, other_time_left)
        else:
            time_left = other_time_left
    if time_left is not None:
        logs.append('time_left:{}s'.format(math.floor(time_left)))

    # join log string and add full metrics report to end of log
    log = '[ {} ] {}'.format(' '.join(logs), train_dict['train_report'])
    print(log)
    train_dict['log_time'].reset()
    return world, agent, train_dict


def __intermediate_validation(opt, valid_world, agent, input_train_dict):
    train_dict = copy.deepcopy(input_train_dict)
    if 0 < opt['validation_every_n_secs'] < train_dict['validate_time'].time() or \
            (opt['validation_every_n_epochs'] > 0 and train_dict['new_epoch'] and (
                    train_dict['epochs_done'] % opt['validation_every_n_epochs']) == 0):

        iopt = copy.deepcopy(opt)
        if iopt.get('evaltask'):
            iopt['task'] = iopt['evaltask']
            print(iopt['task'])
        iopt['datatype'] = 'valid'
        ivalid_world = create_task(iopt, agent)

        valid_report, valid_world = __evaluate_model(ivalid_world, iopt['batchsize'], 'valid',
                                                     iopt['display_examples'], iopt['validation_max_exs'])

        if train_dict['best_metrics'] not in valid_report and 'accuracy' in valid_report:
            train_dict['best_metrics'] = 'accuracy'
        if valid_report[train_dict['best_metrics']] > train_dict['best_metrics_value']:
            train_dict['best_metrics_value'] = valid_report[train_dict['best_metrics']]
            train_dict['impatience'] = 0
            train_dict['lr_drop_impatience'] = 0
            print('[ new best ' + train_dict['best_metrics'] + ': ' + str(train_dict['best_metrics_value']) + ' ]')
            valid_world.save_agents()
            train_dict['saved'] = True
        else:
            train_dict['impatience'] += 1
            train_dict['lr_drop_impatience'] += 1
            print('[ did not beat best ' + train_dict['best_metrics'] + ': {} impatience: {} ]'.format(
                round(train_dict['best_metrics_value'], 4), train_dict['impatience']))
        train_dict['validate_time'].reset()
        if 0 < opt['validation_patience'] <= train_dict['impatience']:
            print('[ ran out of patience! stopping training. ]')
            train_dict['break'] = True
        if 'lr_drop_patience' in opt and 0 < opt['lr_drop_patience'] <= train_dict['lr_drop_impatience']:
            if hasattr(agent, 'drop_lr'):
                print('[ validation metric is decreasing, dropping learning rate ]')
                train_dict['train_report'] = agent.drop_lr()
                agent.reset_metrics()
            else:
                print('[ there is no drop_lr method in agent, ignoring ]')
    return valid_world, agent, train_dict


def __train_single_model(opt):
    """Train single model.
    opt is a dictionary returned by arg_parse
    """
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)
    print('[ training... ]')

    train_dict = {'train_time': Timer(),
                  'validate_time': Timer(),
                  'log_time': Timer(),
                  'new_epoch': None,
                  'epochs_done': 0,
                  'max_exs': opt['num_epochs'] * len(world),
                  'total_exs': 0,
                  'parleys': 0,
                  'max_parleys': math.ceil(opt['num_epochs'] * len(world) / opt['batchsize']),
                  'best_metrics': opt['chosen_metrics'],
                  'best_metrics_value': 0,
                  'impatience': 0,
                  'lr_drop_impatience': 0,
                  'saved': False,
                  'train_report': None,
                  'train_report_agent': None,
                  'train_report_world': None,
                  'break': None}

    try:
        while True:
            world.parley()
            train_dict['parleys'] += 1
            train_dict['new_epoch'] = world.epoch_done()
            if train_dict['new_epoch']:
                world.reset()
                train_dict['epochs_done'] += 1
            if opt['num_epochs'] > 0 and train_dict['parleys'] >= train_dict['max_parleys']:
                print('[ num_epochs completed: {} ]'.format(opt['num_epochs']))
                break
            if 0 < opt['max_train_time'] < train_dict['train_time'].time():
                print('[ max_train_time elapsed: {} ]'.format(train_dict['train_time'].time()))
                break
            world, agent, train_dict = __train_log(opt, world, agent, train_dict)
            _, agent, train_dict = __intermediate_validation(opt, world, agent, train_dict)

            if train_dict['break']:
                break
    except KeyboardInterrupt:
        print('Stopped training, starting testing')

    if not train_dict['saved']:
        world.save_agents()

    world.shutdown()
    agent.shutdown()

    # reload best validation model
    vopt = copy.deepcopy(opt)
    if vopt.get('evaltask'):
        vopt['task'] = vopt['evaltask']
    vopt['datatype'] = 'valid'
    vopt['pretrained_model'] = vopt['model_file']
    agent = create_agent(vopt)
    valid_world = create_task(vopt, agent)
    metrics, _ = __evaluate_model(valid_world, vopt['batchsize'], 'valid',
                                  vopt['display_examples'], vopt['validation_max_exs'])
    valid_world.shutdown()
    agent.shutdown()
    return metrics


def __create_ensemble_model(opt):
    """Create a (set of) model(s).
    opt is a dictionary returned by arg_parse
    """
    metrics_list = []
    folds = opt['bagging_folds_number']
    print('The number of folds is', folds)
    for fold in range(folds):
        print('The {} fold is being trained'.format(fold + 1))
        local_opt = copy.deepcopy(opt)
        local_opt['model_file'] = opt.get('model_file', '') + '_' + str(fold)
        local_opt['bagging_fold_index'] = fold
        metrics_list.append(__train_single_model(local_opt))
    return metrics_list


def model(args=None):
    """Main function.
    args could be provided instead of sys.argv
    """
    args = args if args else sys.argv
    opt = arg_parse(args)

    # Possibly build a dictionary (not all models do this).
    __build_bag_of_words(opt)

    if opt.get('model_files') and opt.get('bagging_folds_number', 0) > 1 and opt.get('bagging_fold_index') is None:
        opt['model_files'] = ['{}_{}'.format(fname, i) for fname in opt['model_files'] for i in range(opt['bagging_folds_number'])]

    if opt['datatype'].split(':')[0] == 'train':
        if opt.get('bagging_folds_number', 0) > 1 and opt.get('bagging_fold_index') is None:
            return __create_ensemble_model(opt)
        else:
            return __train_single_model(opt)
    elif opt['datatype'].split(':')[0] == 'test':
        if opt.get('evaltask'):
            opt['task'] = opt['evaltask']
        agent = create_agent(opt)
        test_world = create_task(opt, agent)
        metrics, _ = __evaluate_model(test_world, opt['batchsize'], 'test',
                                      opt['display_examples'], opt['validation_max_exs'])
        test_world.shutdown()
        agent.shutdown()
        return metrics
    else:
        raise ValueError('--datatype error, please specify "train:..." or "test:..."')


if __name__ == '__main__':
    model()
