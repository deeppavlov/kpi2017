# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    tasks.ner

    This teacher prepares data from Gareev corpus [1].

    Data preprocessing
        1. Every document in Gareev corpus is split into sentences. And every sentence is split into tokens. Then
        all sentences are shuffled
        2. Gareev corpus does not provide train-dev-test split, so the splits are preformed in the teacher. The default
        amounts of data for train, dev, and test are 0.8, 0.1, 0.1 respectively.

    Learning process
        1. The data are fed to the model sentence by sentence.
        2. These sentences are used to form a batch in the model.

    Evaluating
        For evaluation official CoNLL-2003 [2] scorer script is used.
        Target score: CoNLL-2003-F-1.
    References
        [1] https://www.researchgate.net/publication/262203599_Introducing_Baselines_for_Russian_Named_Entity_Recognition
        [2] https://github.com/Franck-Dernoncourt/NeuroNER/blob/master/src/conlleval
"""