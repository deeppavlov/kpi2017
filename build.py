# to run model training type 'pyb train_<task>' replacing <task> with the model name

from pybuilder.core import use_plugin, init, task

use_plugin('python.core')
use_plugin('python.unittest')
use_plugin('python.install_dependencies')

default_task = 'build'

@init
def set_properties(project):
	import os
	import sys
	cwd = os.getcwd()
	sys.path.append(cwd)
	os.environ['IPAVLOV_FTP'] = 'ftp://share.ipavlov.mipt.ru'
	os.environ['EMBEDDINGS_URL'] = 'http://share.ipavlov.mipt.ru:8080/repository/embeddings/'
	os.environ['MODELS_URL'] = 'http://share.ipavlov.mipt.ru:8080/repository/models/'
	os.environ['DATASETS_URL'] = 'http://share.ipavlov.mipt.ru:8080/repository/datasets/'
	project.set_property('dir_source_main_python', '.')
	project.set_property('dir_source_unittest_python', 'tests')


@task
def build(project):
	pass

@task
def train_paraphraser(project):
	import utils.parlai_stuff as ps
	metrics = ps.model(['-t', 'deeppavlov.tasks.paraphrases.agents',
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
                         '--bag-seed', '50',
                         '--bags-per-model', '5',
                         '--validation-patience', '3',
                         '--chosen-metrics', 'f1'
	])

	return metrics

@task
def train_ner(project):
    import utils.parlai_stuff as ps

    metrics = ps.model(['-t', 'deeppavlov.tasks.ner.agents',
                        '-m', 'deeppavlov.agents.ner.ner:NERAgent',
                        '-mf', './build/ner/ner',
                        '-dt', 'train:ordered',
                        '--learning_rate', '0.01',
                        '--batchsize', '2',
                        '--display-examples', 'False',
                        '--max-train-time', '-1', \
                        '--validation-every-n-epochs', '5',
                        '--log-every-n-epochs', '1',
                        '--log-every-n-secs', '-1',
                        '--chosen-metrics', 'f1'
                        ])
    return metrics

@task
def train_insults(project):
    import utils.parlai_stuff as ps
    m1 = ps.model(['-t', 'deeppavlov.tasks.insults.agents:FullTeacher',
                   '-m', 'deeppavlov.agents.insults.insults_agents:OneEpochAgent',
                   '--model_file', './build/insults/log_reg',
                   '-dt', 'train:ordered',
                   '--model_name', 'log_reg',
                   '--log-every-n-secs', '10',
                   '--log-every-n-epochs', '1',
                   '--validation-every-n-epochs', '2',
                   '--raw-dataset-path', './build/insults/',
                   '--batchsize', '64',
                   '--display-examples', 'False',
                   '--max-train-time', '-1',
                   '--num-epochs', '1'
                   ])
    m2 = ps.model(['-t', 'deeppavlov.tasks.insults.agents:FullTeacher',
                   '-m', 'deeppavlov.agents.insults.insults_agents:OneEpochAgent',
                   '--model_file', './build/insults/svc',
                   '-dt', 'train:ordered',
                   '--model_name', 'svc',
                   '--log-every-n-secs', '10',
                   '--log-every-n-epochs', '1',
                   '--validation-every-n-epochs', '2',
                   '--raw-dataset-path', './build/insults/',
                   '--batchsize', '64',
                   '--display-examples', 'False',
                   '--max-train-time', '-1',
                   '--num-epochs', '1'
                   ])
    m3 = ps.model(['-t', 'deeppavlov.tasks.insults.agents',
                   '-m', 'deeppavlov.agents.insults.insults_agents:InsultsAgent',
                   '--model_file', './build/insults/cnn_word',
                   '-dt', 'train:ordered',
                   '--model_name', 'cnn_word',
                   '--log-every-n-secs', '30',
                   '--raw-dataset-path', './build/insults/',
                   '--batchsize', '64',
                   '--display-examples', 'False',
                   '--max-train-time', '-1',
                   '--num-epochs', '50', \
                   '--max_sequence_length', '100',
                   '--learning_rate', '0.01',
                   '--learning_decay', '0.1',
                   '--filters_cnn', '256',
                   '--embedding_dim', '100',
                   '--kernel_sizes_cnn', '3 3 3'
                                         '--regul_coef_conv', '0.001',
                   '--regul_coef_dense', '0.001',
                   '--pool_sizes_cnn', '2 2 2',
                   '--dropout_rate', '0.5',
                   '--dense_dim', '100',
                   '--fasttext_model', './build/insults/reddit_fasttext_model.bin',
                   '--fasttext_embeddings_dict', './build/insults/emb_dict.emb',
                   '--bags-per-model', '3'
                   ])
    m4 = ps.model(['-t', 'deeppavlov.tasks.insults.agents',
                   '-m', 'deeppavlov.agents.insults.insults_agents:InsultsAgent',
                   '--model_file', './build/insults/lstm_word',
                   '-dt', 'train:ordered',
                   '--model_name', 'lstm_word',
                   '--log-every-n-secs', '30',
                   '--raw-dataset-path', './build/insults/',
                   '--batchsize', '64',
                   '--display-examples', 'False',
                   '--max-train-time', '-1',
                   '--num-epochs', '50',
                   '--max_sequence_length', '100',
                   '--learning_rate', '0.01',
                   '--learning_decay', '0.1',
                   '--units_lstm', '128',
                   '--embedding_dim', '100',
                   '--regul_coef_lstm', '0.001'
                                        '--regul_coef_dense', '0.001',
                   '--dropout_rate', '0.5',
                   '--dense_dim', '100',
                   '--fasttext_model', './build/insults/reddit_fasttext_model.bin',
                   '--fasttext_embeddings_dict', './build/insults/emb_dict.emb',
                   '--bags-per-model', '3'
                   ])
    return [m1, m2, m3, m4]

@task
def train_squad(project):
    import utils.parlai_stuff as ps
    metrics = ps.model(['-t', 'squad',
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
                        '--lr_drop', '0.3',
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
