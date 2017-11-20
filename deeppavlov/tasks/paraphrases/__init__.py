
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
Description of the module tasks.paraphraser:

The task of the module

To provide train and test data for the agents.paraphraser module. For this the dataset with paraphrases in russian
posted on the site [1] is used. The module downloads train and test parts of the data. Then it shuffles, splits into
batches and feeds the to the agents.paraphraser module for training (feeds the whole dataset for testing). The part of
the data is reserved for the validation. Namely, the data is divided into k folds, where k equals to the
'--bagging-folds-number' parameter. Each fold is then used once as a validation while the k - 1 remaining folds form the
training set.

[1] http://paraphraser.ru/
"""
