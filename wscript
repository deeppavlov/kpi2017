import os

def train_paraphraser_model(ctx):
  pass

def test_paraphraser_model(ctx):
  pass


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
