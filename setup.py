from setuptools import setup, find_packages
from pip.req import parse_requirements
from pip.download import PipSession
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def read_requirements():
    # parses requirements from requirements.txt
    reqs_path = os.path.join(__location__, 'requirements.txt')
    install_reqs = parse_requirements(reqs_path, session=PipSession())
    reqs = [str(ir.req) for ir in install_reqs]
    return reqs

setup(name='deeppavlov',
      version='0.0.1',
      description='Open source NLP framework',
      url='https://github.com/deepmipt/deeppavlov',
      author='Neural Networks and Deep Learning lab, MIPT',
      author_email='deeppavlov@ipavlov.ai',
      license='Apache License, Version 2.0',
      packages=find_packages(exclude=('data', 'docs', 'downloads', 'utils', 'logs', 'tests', 'src')),
      include_package_data=True,
      install_requires=read_requirements(),
      keywords=['NLP',
                'natural language processing',
                'paraphrase',
                'NER',
                'named entity recognition',
                'coreference resolution'],
      )
