import unittest
import build_utils as bu
import datetime


class KPIException(Exception):
    """Class for exceptions raised if some KPI is not satisfied"""
    pass


class TestKPIs(unittest.TestCase):
    """Class for tests of different KPIs"""

    report_string = '{:%Y/%m/%d %H:%M} {}: actual {}, expected {}\n'
    report_file = './kpi_score_reports'

    @classmethod
    def report_score(cls, kpi, actual, expected):
        report = cls.report_string.format(datetime.datetime.now(), kpi, actual, expected)
        print(report)
        with open(cls.report_file, 'a+') as f:
            f.write(report)

    def test_paraphraser(self):
        expected_score = 0.8
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
        TestKPIs.report_score("paraphraser", metrics["f1"], expected_score)

        self.assertTrue(metrics['f1'] > expected_score,
                        'KPI for paraphraser is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], expected_score))

    def test_ner(self):
        expected_score = 70
        metrics = bu.model(['-t', 'deeppavlov.tasks.ner.agents',
                            '-m', 'deeppavlov.agents.ner.ner:NERAgent',
                            '-mf', './build/ner/ner',
                            '-dt', 'test',
                            '--batchsize', '2',
                            '--display-examples', 'False',
                            '--validation-every-n-epochs', '5',
                            '--log-every-n-epochs', '1',
                            '--log-every-n-secs', '-1',
                            '--pretrained-model', './build/ner/ner',
                            '--chosen-metrics', 'f1'
                            ])

        TestKPIs.report_score("ner", metrics["f1"], expected_score)

        self.assertTrue(metrics['f1'] > expected_score,
                        'KPI for NER is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], expected_score))

    def test_insults(self):
        expected_score = 0.85
        metrics = bu.model(['-t', 'deeppavlov.tasks.insults.agents:FullTeacher',
                            '-m', 'deeppavlov.agents.insults.insults_agents:EnsembleInsultsAgent',
                            '--model_file', './build/insults/insults_ensemble',
                            '--model_files', './build/insults/cnn_word_0',
                            './build/insults/cnn_word_1',
                            './build/insults/cnn_word_2',
                            '--model_names', 'cnn_word',
                            'cnn_word', 'cnn_word',
                            '--model_coefs', '0.3333333',
                            '0.3333333', '0.3333334',
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

        TestKPIs.report_score("insults", metrics["auc"], expected_score)

        self.assertTrue(metrics['auc'] > expected_score,
                        'KPI for insults is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['auc'], expected_score))

    def test_squad(self):
        expected_score = 0.7
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

        TestKPIs.report_score("SQuAD", metrics["f1"], expected_score)

        self.assertTrue(metrics['f1'] > expected_score,
                        'KPI for SQuAD is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], expected_score))
    
    """
    def test_coreference(self):
        expected_score = 0.55
        metrics = bu.model(['-t', 'deeppavlov.tasks.coreference.agents',
                            '-m', 'deeppavlov.agents.coreference.agents:CoreferenceAgent',
                            '-mf', './build/coreference/',
                            '-dt', 'test',
                            '--language', 'russian',
                            '--name', 'fasttext',
                            '--pretrained_model', 'True',
                            '--datatype', 'test:stream',
                            '--batchsize', '1',
                            '--display-examples', 'False',
                            '--chosen-metric', 'f1'
                            ])

        TestKPIs.report_score("Coreference", metrics["f1"], expected_score)

        self.assertTrue(metrics['f1'] > expected_score,
                        'KPI for Coreference resolution is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], expected_score))
    """
    
    def test_coreference_scorer_model(self):
        expected_score = 0.55

        metrics = bu.model(['-t', 'deeppavlov.tasks.coreference_scorer_model.agents:CoreferenceTeacher',
                    '-m', 'deeppavlov.agents.coreference_scorer_model.agents:CoreferenceAgent',
                    '--display-examples', 'False',
                    '--num-epochs', '-1',
                    '--log-every-n-secs', '-1',
                    '--log-every-n-epochs', '1',
                    '--validation-every-n-epochs', '-1',
                    '--chosen-metrics', 'f1',
                    '--datatype', 'test',
                    '--model-file', './build/coref',
                    '--pretrained_model', './build/coref',
                    '--embeddings_path', './build/coref/fasttext_embdgs.bin',
                    ])

        TestKPIs.report_score("Coreference Scorer Model", metrics["f1"], expected_score)

        self.assertTrue(metrics['f1'] > expected_score,
                        'KPI for Coreference resolution is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], expected_score))


if __name__ == '__main__':
    unittest.main()

