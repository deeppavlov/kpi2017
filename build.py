# to run model training type 'pyb train_<task>' replacing <task> with the model name

from pybuilder.core import use_plugin, init, task
import os
import build_utils as bu

use_plugin('python.core')
use_plugin('python.unittest')
use_plugin('python.install_dependencies')

default_task = 'build'


@init
def set_properties(project):
    import sys
    if not os.path.exists('./build'):
        os.mkdir('build', mode=0o755)
    cwd = os.getcwd()
    sys.path.append(cwd)
    os.environ['EMBEDDINGS_URL'] = os.getenv('EMBEDDINGS_URL',
                                             default='http://share.ipavlov.mipt.ru:8080/repository/embeddings/')
    os.environ['MODELS_URL'] = os.getenv('MODELS_URL',
                                         default='http://share.ipavlov.mipt.ru:8080/repository/models/')
    os.environ['DATASETS_URL'] = os.getenv('DATASETS_URL',
                                           default='http://share.ipavlov.mipt.ru:8080/repository/datasets/')
    os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('CUDA_VISIBLE_DEVICES',
                                                   default='7')
    project.set_property('dir_source_main_python', '.')
    project.set_property('dir_source_unittest_python', 'tests')


@task
def build(project):
    pass


@task
def clean(project):
    import shutil
    shutil.rmtree('./build')


@task
def train_paraphraser(project):
    if not os.path.exists('./build/paraphraser'):
        os.mkdir('build/paraphraser', mode=0o755)
    metrics = bu.model(['-t', 'deeppavlov.tasks.paraphrases.agents',
                        '-m', 'deeppavlov.agents.paraphraser.paraphraser:ParaphraserAgent',
                        '-mf', './build/paraphraser/paraphraser',
                        '--datatype', 'train:ordered',
                        '--batchsize', '256',
                        '--display-examples', 'False',
                        '--max-train-time', '-1',
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
    if not os.path.exists('./build/ner'):
        os.mkdir('build/ner', mode=0o755)

    metrics = bu.model(['-t', 'deeppavlov.tasks.ner.agents',
                        '-m', 'deeppavlov.agents.ner.ner:NERAgent',
                        '-mf', './build/ner/ner',
                        '-dt', 'train:ordered',
                        '--learning_rate', '0.01',
                        '--batchsize', '2',
                        '--display-examples', 'False',
                        '--max-train-time', '-1',
                        '--validation-every-n-epochs', '5',
                        '--log-every-n-epochs', '1',
                        '--log-every-n-secs', '-1',
                        '--chosen-metrics', 'f1'
                        ])
    return metrics


@task
def train_insults(project):
    if not os.path.exists('./build/insults'):
        os.mkdir('build/insults', mode=0o755)
    metrics = bu.model(['-t', 'deeppavlov.tasks.insults.agents',
                        '-m', 'deeppavlov.agents.insults.insults_agents:InsultsAgent',
                        '--model_file', './build/insults/cnn_word',
                        '-dt', 'train:ordered',
                        '--model_name', 'cnn_word',
                        '--log-every-n-secs', '60',
                        '--raw-dataset-path', './build/insults/',
                        '--batchsize', '64',
                        '--display-examples', 'False',
                        '--max-train-time', '-1',
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
                        '-vp', '1'
                        ])
    return metrics


@task
def train_squad(project):
    if not os.path.exists('./build/squad'):
        os.mkdir('build/squad', mode=0o755)
    metrics = bu.model(['-t', 'squad',
                        '-m', 'deeppavlov.agents.squad.squad:SquadAgent',
                        '--batchsize', '64',
                        '--display-examples', 'False',
                        '--max-train-time', '-1',
                        '--num-epochs', '-1',
                        '--log-every-n-secs', '60',
                        '--log-every-n-epochs', '-1',
                        '--validation-every-n-secs', '1800',
                        '--validation-every-n-epochs', '-1',
                        '--chosen-metrics', 'f1',
                        '--validation-patience', '5',
                        '--lr-drop-patience', '1',
                        '--type', 'fastqa_default',
                        '--lr', '0.0001',
                        '--linear_dropout', '0.0',
                        '--embedding_dropout', '0.5',
                        '--rnn_dropout', '0.0',
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
