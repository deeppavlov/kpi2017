import unittest
import build_utils as bu
import datetime


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    if loader.testMethodPrefix != 'test':
        test_cases = loader.testMethodPrefix.split(',')
        for i in range(len(test_cases)):
            test_cases[i] = __name__ + '.' + ''.join([word.capitalize() for word in test_cases[i].split('_')])
        loader.testMethodPrefix = 'test' #return to default
        tests = loader.loadTestsFromNames(test_cases)
    suite.addTests(tests)
    return suite


class TestModels(unittest.TestCase):
    """Parent class for model tests"""

    report_string = '{:%Y/%m/%d %H:%M} {}: actual {}, expected {}\n'
    report_file = './build/kpi_score_reports.txt'

    @classmethod
    def report_score(cls, kpi, actual, expected):
        report = cls.report_string.format(datetime.datetime.now(), kpi, actual, expected)
        print(report)
        with open(cls.report_file, 'a+') as f:
            f.write(report)


class TestParaphraser(TestModels):
    expected_score = 0.8
    def test_KPI(self):
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
        TestModels.report_score("paraphraser", metrics["f1"], self.expected_score)

        self.assertTrue(metrics['f1'] > self.expected_score,
                        'KPI for paraphraser is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], self.expected_score))


class TestIdleParaphraser(TestParaphraser):
    expected_score = 0


class TestNer(TestModels):
    expected_score = 70.0
    def test_KPI(self):
        metrics = bu.model(['-t', 'deeppavlov.tasks.ner.agents',
                            '-m', 'deeppavlov.agents.ner.ner:NERAgent',
                            '-mf', './build/ner',
                            '-dt', 'test',
                            '--dict-file', './build/ner/dict',
                            '--batchsize', '2',
                            '--display-examples', 'False',
                            '--validation-every-n-epochs', '5',
                            '--log-every-n-epochs', '1',
                            '--log-every-n-secs', '-1',
                            '--pretrained-model', './build/ner',
                            '--chosen-metrics', 'f1'
                            ])

        TestModels.report_score("ner", metrics["f1"], self.expected_score)

        self.assertTrue(metrics['f1'] > self.expected_score,
                        'KPI for NER is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], self.expected_score))


class TestIdleNer(TestNer):
    expected_score = 0


class TestInsults(TestModels):
    expected_score = 0.85
    def test_KPI(self):
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

        TestModels.report_score("insults", metrics["auc"], self.expected_score)

        self.assertTrue(metrics['auc'] > self.expected_score,
                        'KPI for insults is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['auc'], self.expected_score))


class TestIdleInsults(TestInsults):
    expected_score = 0


class TestSquad(TestModels):
    expected_score = 0.7
    def test_KPI(self):
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

        TestModels.report_score("SQuAD", metrics["f1"], self.expected_score)

        self.assertTrue(metrics['f1'] > self.expected_score,
                        'KPI for SQuAD is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], self.expected_score))


class TestIdleSquad(TestSquad):
    expected_score = 0


class TestCoreference(TestModels):
    expected_score = 0.55
    def test_KPI(self):
        metrics = bu.model(['-t', 'deeppavlov.tasks.coreference.agents',
                            '-m', 'deeppavlov.agents.coreference.agents:CoreferenceAgent',
                            '-mf', './build/coreference/',
                            '--language', 'russian',
                            '--name', 'gold_main',
                            '--pretrained_model', 'True',
                            '--datatype', 'test:stream',
                            '--batchsize', '1',
                            '--display-examples', 'False',
                            '--chosen-metric', 'conll-F-1',
                            '--train_on_gold', 'True',
                            '--random_seed', '5'
                            ])

        TestModels.report_score("Coreference", metrics["conll-F-1"], self.expected_score)

        self.assertTrue(metrics['conll-F-1'] > self.expected_score,
                        'KPI for Coreference resolution is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['conll-F-1'], self.expected_score))


class TestIdleCoreference(TestCoreference):
    expected_score = 0


class TestCoref(TestModels):
    expected_score = 0.55
    def test_KPI(self):
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

        TestModels.report_score("Coreference Scorer Model", metrics["f1"], self.expected_score)

        self.assertTrue(metrics['f1'] > self.expected_score,
                        'KPI for Coreference resolution is not satisfied. \
                        Got {}, expected more than {}'.format(metrics['f1'], self.expected_score))


class TestIdleCoref(TestCoref):
    expected_score = 0


if __name__ == '__main__':
    unittest.main()

