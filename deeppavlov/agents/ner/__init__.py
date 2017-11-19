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
    agents.ner
    This agent can be used for Named Entity Recognition (NER).
    NER task can be treated as per token classification. For each token the network must provide a tag from predefined
    set of tags. The common types of tags are: persons, organizations, locations, etc. BIO markup is used for
    distinguishing consequent entities. In this agent convolutional neural network is used for solving NER task [1].

    Model architecture:
        1. The input of the model is utterance. Utterance is split into tokens. For each token token level and character
        level embeddings are used. Token level embeddings are produced by lookup table (namely matrix, which rows are
        distributed representations of the tokens). Character level representations are produced by character level
        convolutional neural network (CNN). The input to character level CNN is character embeddings.

        2. Token level convolutional neural network for tagging. The embeddings from character and token level are
        concatenated and fed into the tagging network. The output of the tagging token level CNN is passed on Dense
        layer (actually a number of layers with shared weights) to produce probability distribution over the possible
        tags.S

    Details
        The tags for data must be presented in BIO format. Embeddings are trained on the go. The metric used for
        scorring is F1. However precision and recall sometimes are used as well. For F1 score calculation CoNLL 2003
        script is used. It calculates weighted score for the total set of tags. If tagging of a specific entity is
        different even in 1 tag it counts as error.


    References
        [1] http://www.cips-cl.org/static/anthology/CCL-2017/CCL-17-071.pdf

"""