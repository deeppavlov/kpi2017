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
                         '-mf', './build/paraphraser', \
                         '--model_files', './build/paraphraser', \
                         '--datatype', 'test', \
                         '--batchsize', '256', \
                         '--display-examples', 'False', \
                         '--fasttext_embeddings_dict', './build/paraphraser.emb', \
                         '--fasttext_model', './build/ft_0.8.3_nltk_yalen_sg_300.bin', \
                         '--cross-validation-splits-count', '5', \
                         '--chosen-metric', 'f1'
	])
    
        self.assertTrue(metrics['f1'] > 0.8, 'KPI for paraphraser is not satisfied')

    def test_ner(self):
        pass

if __name__ == '__main__':
    unittest.main()

