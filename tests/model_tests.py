import unittest
from utils import train_model


class KPIException(Exception):
    """Class for exceptions raised if some KPI is not satisfied"""
    pass


class KPITests(unittest.TestCase):
    """Class for tests of different KPIs"""


    def test_paraphraser(self):
        metrics = train_model.main(['-t', 'deeppavlov.tasks.paraphrases.agents', \
                         '-m', 'deeppavlov.agents.paraphraser.paraphraser:EnsembleParaphraserAgent', \
                         '-mf', './build/paraphraser/paraphraser', \
                         '--model_files', './build/paraphraser/paraphraser', \
                         '--datatype', 'test', \
                         '--batchsize', '256', \
                         '--display-examples', 'False', \
                         '--fasttext_embeddings_dict', './build/paraphraser/paraphraser.emb', \
                         '--fasttext_model', './build/paraphraser/ft_0.8.3_nltk_yalen_sg_300.bin', \
                         '--cross-validation-splits-count', '5', \
                         '--chosen-metric', 'f1'
	])
        self.assertTrue(metrics['f1'] > 0.8, 'KPI for paraphraser is not satisfied')


    def test_ner(self):
	metrics = utils.train_model.main(['-t', 'deeppavlov.tasks.ner.agents', \
                         '-m', 'deeppavlov.agents.ner.ner:NERAgent', \
                         '-mf', './build/ner', \
                         '-dt', 'test', \
                         '--batchsize', '2', \
                         '--display-examples', 'False', \
                         '--max-train-time', '-1', \
                         '--validation-every-n-epochs', '5', \
                         '--log-every-n-epochs', '1', \
                         '--log-every-n-secs', '-1',  \
                         '--pretrained-model', './build/ner',  \
                         '--chosen-metric', 'f1'
	])
        self.assertTrue(metrics['f1'] > 0.7, 'KPI for NER is not satisfied')


    def test_insults(self):
        metrics = train_model.main(['-t', 'deeppavlov.tasks.paraphrases.agents', \
                         '-m', 'deeppavlov.agents.insults.insults_agents:EnsembleInsultsAgent', \
                               '--model_file', './build/insults/insults_ensemble' \
#                               '--model_files', './build/insults/cnn_word_0 \
#                                             ./build/insults/cnn_word_1 \
#                                             ./build/insults/cnn_word_2 \
#                                             ./build/insults/lstm_word_0 \
#                                             ./build/insults/lstm_word_1 \
#                                             ./build/insults/lstm_word_2 \
#                                             ./build/insults/log_reg \
#                                             ./build/insults/svc', \
                               '--model_files', './build/insults/cnn_word_0 \
                                             ./build/insults/cnn_word_1 \
                                             ./build/insults/cnn_word_2 \
                                             ./build/insults/log_reg \
                                             ./build/insults/svc', \
                               '--model_names', 'cnn_word cnn_word cnn_word lstm_word lstm_word lstm_word log_reg svc', \
                               '--model_coefs', '0.05 0.05 0.05 0.05 0.05 0.05 0.2 0.5', \
                               '--datatype', 'test', \
                               '--batchsize', '64', \
                               '--display-examples', 'False', \
                               '--raw-dataset-path', './build/insults/', \
                               '--max-train-time', '-1', \
                               '--num-epochs', '1', \
                               '--max_sequence_length', '100', \
                               '--learning_rate', '0.01', \
                               '--learning_decay', '0.1', \
                               '--filters_cnn', '256', \
                               '--kernel_sizes_cnn', '3 3 3', \
                               '--regul_coef_conv', '0.001', \
                               '--regul_coef_dense', '0.001', \
                               '--pool_sizes_cnn', '2 2 2',  \
                               '--units_lstm', '128' \
                               '--embedding_dim', '100' \
                               '--regul_coef_lstm', '0.001', \
                               '--dropout_rate', '0.5' \
                               '--dense_dim', '100', \
                               '--fasttext_model', './build/insults/reddit_fasttext_model.bin'
	])
        self.assertTrue(metrics['auc'] > 0.85, 'KPI for insults is not satisfied')	


    def test_squad(self):
        metrics = train_model.main(['-t', 'squad', \
                         'm', 'deeppavlov.agents.squad.squad:SquadAgent', \
                         '--batchsize', '64', \
                         '--display-examples', 'False', \
                         '--max-train-time', '-1', \
                         '--num-epochs', '-1', \
                         '--log-every-n-secs', '60', \
                         '--log-every-n-epochs', '-1', \
                         '--validation-every-n-secs', '1800', \
                         '--validation-every-n-epochs', '-1', \
                         '--chosen-metric', 'f1', \
                         '--validation-patience', '5', \
                         '--lr-drop-patience', '1', \
                         '--type', 'fastqa_default', \
                         '--lr', '0.0001', \
                         '--lr_drop', '0.3', \
                         '--linear_dropout', '0.0', \
                         '--embedding_dropout', '0.5', \
                         '--rnn_dropout', '0.0', \
                         '--recurrent_dropout', '0.0', \
                         '--input_dropout', '0.0', \
                         '--output_dropout', '0.0', \
                         '--context_enc_layers', '1', \
                         '--question_enc_layers', '1', \
                         '--encoder_hidden_dim', '300', \
                         '--projection_dim', '300', \
                         '--pointer_dim', '300', \
                         '--model-file', './build/squad/squad1', \
                         '--embedding_file', './build/squad/glove.840B.300d.txt', \
                         '--pretrained_model', './build/squad/squad1', \
                         '--datatype', 'test'
	])
        self.assertTrue(metrics['f1'] > 0.7, 'KPI for SQuAD is not satisfied')


if __name__ == '__main__':
    unittest.main()

