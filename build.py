from pybuilder.core import use_plugin, init, task
import os
import build_utils as bu

use_plugin('python.core')
use_plugin('python.unittest')
use_plugin('python.install_dependencies')

default_task = 'build'


def create_dir(dir):
    os.makedirs('build/' + dir, mode=0o755, exist_ok=True)


@init
def set_properties(project):
    import sys
    cwd = os.getcwd()
    sys.path.append(cwd)
    os.environ['EMBEDDINGS_URL'] = os.getenv('EMBEDDINGS_URL',
                                             default='http://share.ipavlov.mipt.ru:8080/repository/embeddings/')
    os.environ['MODELS_URL'] = os.getenv('MODELS_URL',
                                         default='http://share.ipavlov.mipt.ru:8080/repository/models/')
    os.environ['DATASETS_URL'] = os.getenv('DATASETS_URL',
                                           default='http://share.ipavlov.mipt.ru:8080/repository/datasets/')
    os.environ['KERAS_BACKEND'] = os.getenv('KERAS_BACKEND', default='tensorflow')
    project.set_property('dir_source_main_python', '.')
    project.set_property('dir_source_unittest_python', 'tests')


@task
def build(project):
    pass


@task
def clean(project):
    import shutil
    shutil.rmtree('./build', ignore_errors=True)


@task
def train_paraphraser(project):
    create_dir('paraphraser')
    metrics = bu.model(['-t', 'deeppavlov.tasks.paraphrases.agents',
                        '-m', 'deeppavlov.agents.paraphraser.paraphraser:ParaphraserAgent',
                        '-mf', './build/paraphraser/paraphraser',
                        '--datatype', 'train:ordered',
                        '--batchsize', '256',
                        '--display-examples', 'False',
                        '--num-epochs', '-1',
                        '--log-every-n-secs', '-1',
                        '--log-every-n-epochs', '1',
                        '--learning_rate', '0.0001',
                        '--hidden_dim', '200',
                        '--validation-every-n-epochs', '5',
                        '--fasttext_embeddings_dict', './build/paraphraser/paraphraser.emb',
                        '--fasttext_model', './build/paraphraser/ft_0.8.3_nltk_yalen_sg_300.bin',
                        '--teacher-random-seed', '50',
                        '--bagging-folds-number', '5',
                        '--validation-patience', '3',
                        '--chosen-metrics', 'f1'
                        ])
    return metrics


@task
def train_ner(project):
    create_dir('ner')
    metrics = bu.model(['-t', 'deeppavlov.tasks.ner.agents',
                        '-m', 'deeppavlov.agents.ner.ner:NERAgent',
                        '-mf', './build/ner',
                        '-dt', 'train:ordered',
                        '--dict-file', './build/ner/dict',
                        '--learning_rate', '0.01',
                        '--batchsize', '2',
                        '--display-examples', 'False',
                        '--validation-every-n-epochs', '5',
                        '--log-every-n-epochs', '1',
                        '--log-every-n-secs', '-1',
                        '--chosen-metrics', 'f1'
                        ])
    return metrics


@task
def train_insults(project):
    create_dir('insults')
    metrics = bu.model(['-t', 'deeppavlov.tasks.insults.agents',
                        '-m', 'deeppavlov.agents.insults.insults_agents:InsultsAgent',
                        '--model_file', './build/insults/cnn_word',
                        '-dt', 'train:ordered',
                        '--model_name', 'cnn_word',
                        '--log-every-n-secs', '60',
                        '--raw-dataset-path', './build/insults/',
                        '--batchsize', '64',
                        '--display-examples', 'False',
                        '--num-epochs', '1000',
                        '--max_sequence_length', '100',
                        '--learning_rate', '0.01',
                        '--learning_decay', '0.1',
                        '--filters_cnn', '256',
                        '--embedding_dim', '100',
                        '--kernel_sizes_cnn', '1 2 3',
                        '--regul_coef_conv', '0.001',
                        '--regul_coef_dense', '0.01',
                        '--dropout_rate', '0.5',
                        '--dense_dim', '100',
                        '--fasttext_model', './build/insults/reddit_fasttext_model.bin',
                        '--fasttext_embeddings_dict', './build/insults/emb_dict.emb',
                        '--bagging-folds-number', '3',
                        '-ve', '10',
                        '-vp', '5',
                        '--chosen-metric', 'auc'
                        ])
    return metrics


@task
def train_squad(project):
    create_dir('squad')
    metrics = bu.model(['-t', 'squad',
                        '-m', 'deeppavlov.agents.squad.squad:SquadAgent',
                        '--batchsize', '64',
                        '--display-examples', 'False',
                        '--num-epochs', '-1',
                        '--log-every-n-secs', '60',
                        '--log-every-n-epochs', '-1',
                        '--validation-every-n-secs', '1800',
                        '--validation-every-n-epochs', '-1',
                        '--chosen-metrics', 'f1',
                        '--validation-patience', '5',
                        '--type', 'fastqa_default',
                        '--lr', '0.001',
                        '--lr_drop', '0.3',
                        '--linear_dropout', '0.25',
                        '--embedding_dropout', '0.5',
                        '--rnn_dropout', '0.25',
                        '--recurrent_dropout', '0.0',
                        '--input_dropout', '0.0',
                        '--output_dropout', '0.0',
                        '--context_enc_layers', '1',
                        '--question_enc_layers', '1',
                        '--encoder_hidden_dim', '300',
                        '--projection_dim', '300',
                        '--pointer_dim', '300',
                        '--model-file', './build/squad/squad1',
                        '--embedding_file', './build/squad/glove.840B.300d.txt'
                        ])
    return metrics


def compile_coreference(path):
    path = path + '/coref_kernels.so'
    if not os.path.isfile(path):
        print('Compiling the coref_kernels.cc')
        cmd = """#!/usr/bin/env bash

                # Build custom kernels.
                TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

                # Linux (pip)
                g++ -std=c++11 -shared ./deeppavlov/agents/coreference/coref_kernels.cc -o {0} -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0

                # Linux (build from source)
                #g++ -std=c++11 -shared ./deeppavlov/agents/coreference/coref_kernels.cc -o {0} -I $TF_INC -fPIC

                # Mac
                #g++ -std=c++11 -shared ./deeppavlov/agents/coreference/coref_kernels.cc -o {0} -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0  -undefined dynamic_lookup"""
        cmd = cmd.format(path)
        os.system(cmd)
        print('End of compiling the coref_kernels.cc')


@task
def train_coreference(project):
    create_dir('coreference')
    mf = './build/coreference/'
    compile_coreference(mf)
    metrics = bu.model(['-t', 'deeppavlov.tasks.coreference.agents',
                        '-m', 'deeppavlov.agents.coreference.agents:CoreferenceAgent',
                        '-mf', mf,
                        '--language', 'russian',
                        '--name', 'main',
                        '--pretrained_model', 'False',
                        '-dt', 'train:ordered',
                        '--batchsize', '1',
                        '--display-examples', 'False',
                        '--num-epochs', '1500',
                        '--validation-every-n-epochs', '50',
                        '--nitr', '1500',
                        '--log-every-n-epochs', '1',
                        '--log-every-n-secs', '-1',
                        '--chosen-metric', 'conll-F-1',
                        '--validation-patience', '10'
                        ])
    return metrics


@task
def train_coreference_scorer_model(project):
    create_dir('coref')
    metrics = bu.model(['-t', 'deeppavlov.tasks.coreference_scorer_model.agents:CoreferenceTeacher',
                        '-m', 'deeppavlov.agents.coreference_scorer_model.agents:CoreferenceAgent',
                        '--display-examples', 'False',
                        '--num-epochs', '20',
                        '--log-every-n-secs', '-1',
                        '--log-every-n-epochs', '1',
                        '--validation-every-n-epochs', '1',
                        '--chosen-metrics', 'f1',
                        '--validation-patience', '5',
                        '--model-file', './build/coref',
                        '--embeddings_path', './build/coref/fasttext_embdgs.bin'
                        ])
    return metrics
