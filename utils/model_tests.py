import unittest
import train_model as model

class KPIException(Exception):
    """Class for exceptions raised if some KPI is not satisfied"""
    pass


class KPITests(unittest.TestCase):
    """Class for tests of different KPIs"""

    def test_paraphraser(self):
        metric = model.main(['-t', 'deeppavlov.tasks.paraphrases.agents',
                         '-m', 'deeppavlov.agents.paraphraser.paraphraser:ParaphraserAgent',
                         '-mf', './build/paraphraser',
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
                         '--fasttext_embeddings_dict', '"./build/paraphraser.emb"',
                         '--fasttext_model', '"./build/ft_0.8.3_nltk_yalen_sg_300.bin"',
                         '--cross-validation-seed', '50',
                         '--cross-validation-splits-count', '5',
                         '--validation-patience', '3',
                         '--chosen-metric', 'f1'
#,                         '--pretrained_model', './build/paraphraser'
	])
    
        self.assertTrue(metric['f1'] > 0.8)

    def test_ner(self):
        pass


if __name__ == '__main__':
    unittest.main()
