import os
import utils.train_model as tm

def train_paraphraser_model(ctx):
    tm.main(['-t', 'deeppavlov.tasks.paraphrases.agents',
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
    tm.main(['-t', 'deeppavlov.tasks.paraphrases.agents',
                         '-m', 'deeppavlov.agents.paraphraser.paraphraser:EnsembleParaphraserAgent',
                         '-mf', './build/paraphraser',
                         '--model_files', './build/paraphraser',
                         '--datatype', 'test',
                         '--batchsize', '256'
                         '--display-examples', 'False'
                         '--fasttext_embeddings_dict', "./build/paraphraser.emb"',
                         '--fasttext_model', '"./build/ft_0.8.3_nltk_yalen_sg_300.bin"',
                         '--cross-validation-splits-count', '5'
                         '--chosen-metric', 'f1'
	])
    return

def test_paraphraser_model(ctx):
    
    return


def train_models(ctx):
  pass

def deploy_models_to_nexus(ctx):
  pass

def deploy_library_to_pip(ctx):
  pass

def clean(ctx):
  pass


def configure(ctx):
    ctx.env.PYTHONPATH = os.getcwd()
