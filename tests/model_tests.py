import unittest
from utils import train_model
#from utils.train_paraphraser import train_paraphraser_model as tp

class KPITests(unittest.TestCase):
    """Class for tests of different KPIs"""
    def test_paraphraser(self):
        pass
#        self.assertTrue(tp() > 0.8, 'KPI for paraphraser is not satisfied')
    def test_paraphraser(self):
        metrics = train_model.main(['-t', 'deeppavlov.tasks.paraphrases.agents', \
                         '-m', 'deeppavlov.agents.paraphraser.paraphraser:EnsembleParaphraserAgent', \
                         '-mf', './build/maxpool_match', \
                         '--model_files', './build/maxpool_match', \
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

