import unittest
import build_utils as bu


class KPIException(Exception):
    """Class for exceptions raised if some KPI is not satisfied"""
    pass


class TestKPIs(unittest.TestCase):
    """Class for tests of different KPIs"""


    def test_paraphraser(self):
        metrics = bu.model(['-t', 'deeppavlov.tasks.paraphrases.agents',
                            '-m', 'deeppavlov.agents.paraphraser.paraphraser:EnsembleParaphraserAgent',
                            '-mf', './build/paraphraser/paraphraser',
                            '--model_files', './build/paraphraser/paraphraser',
                            '--datatype', 'test',
                            '--batchsize', '256',
                            '--display-examples', 'False',
                            '--fasttext_embeddings_dict', './build/paraphraser/paraphraser.emb',
                            '--fasttext_model', './build/paraphraser/ft_0.8.3_nltk_yalen_sg_300.bin',
                            '--bagging-folds-number', '5',
                            '--chosen-metrics', 'f1'
                            ])
        print(metrics)
        self.assertTrue(metrics['f1'] > 0.8, 'KPI for paraphraser is not satisfied')

    def test_ner(self):
        metrics = bu.model(['-t', 'deeppavlov.tasks.ner.agents',
                            '-m', 'deeppavlov.agents.ner.ner:NERAgent',
                            '-mf', './build/ner/ner',
                            '-dt', 'test',
                            '--batchsize', '2',
                            '--display-examples', 'False',
                            '--max-train-time', '-1',
                            '--validation-every-n-epochs', '5',
                            '--log-every-n-epochs', '1',
                            '--log-every-n-secs', '-1',
                            '--pretrained-model', './build/ner/ner',
                            '--chosen-metrics', 'f1'
                            ])
        self.assertTrue(metrics['f1'] > 70, 'KPI for NER is not satisfied')

    def test_insults(self):
        metrics = bu.model(['-t', 'deeppavlov.tasks.insults.agents:FullTeacher',
                            '-m', 'deeppavlov.agents.insults.insults_agents:EnsembleInsultsAgent',
                            '--model_file', './build/insults/insults_ensemble',
                            '--model_files', './build/insults/cnn_word_0 \
                                    ./build/insults/cnn_word_1 \
                                    ./build/insults/cnn_word_2',
                            '--model_names', 'cnn_word cnn_word cnn_word',
                            '--model_coefs', '0.3333333 0.3333333 0.3333334',
                            '--datatype', 'test',
                            '--batchsize', '64',
                            '--display-examples', 'False',
                            '--raw-dataset-path', './build/insults/',
                            '--max_sequence_length', '100',
                            '--filters_cnn', '256',
                            '--kernel_sizes_cnn', '1 2 3',
                            '--embedding_dim', '100',
                            '--dense_dim', '100',
                            '--fasttext_model', './build/insults/reddit_fasttext_model.bin'
                            ])
        self.assertTrue(metrics['auc'] > 0.85, 'KPI for insults is not satisfied')

    def test_squad(self):
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
                            '--embedding_file', './build/squad/glove.840B.300d.txt',
                            '--pretrained_model', './build/squad/squad1',
                            '--datatype', 'test'
                            ])
        self.assertTrue(metrics['f1'] > 0.7, 'KPI for SQuAD is not satisfied')


if __name__ == '__main__':
    unittest.main()

