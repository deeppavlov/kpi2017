# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Description of the module agents.paraphraser:

The task of the module

To recognize whether two sentences are paraphrases or not. The module should give
a positive answer in the case if two sentences are paraphrases, and a negative answer in the other case.

Models architecture

All models use Siamese architecture. Word embeddings to models are provided by the pretrained fastText  model.
A sentence level context is taken into account using LSTM or bi-LSTM layer. Most models use attention to identify
similar parts in sentences. Currently implemented types of attention include multiplicative attention [1] and various
types of multi-perspective matching [2]. After the attention layer the absolute value of the difference and element-wise
product of two vectors representing the sentences are calculated. These vectors are concatenated and input to dense
layer with a sigmoid activation performing final classification. The chosen model is trained k times, where k equals to
the '--bagging-folds-number' parameter corresponding to the number of data folds. Predictions of the model trained on
various data subsets are averaged at testing time (bagging). There is a possibility to choose few models for training
at once. Each of them will be trained and their predictions will be averaged at testing time (ensembling).

[1] Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation.
EMNLP 2015. CoRR, abs/1508.04025

[2] Zhiguo Wang, Wael Hamza, & Radu Florian. Bilateral multi-perspective matching for natural language sentences.
CoRR, abs/1702.03814, 2017.
"""

