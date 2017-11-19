# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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
    tasks.insults
    
    The challenge was presented as a competition on Kaggle in 2012.
    Three datasets are given as train (train.csv), valid (test_with_solutions.csv), 
    and test (test.csv, impermium_verification_labels.csv) [1]. 
    The train and validation datasets are combined to obtain more train data,
    cross-validation technique will be used.
        
    Data preprocessing
    
        1. All letters to lower-case.
        2. Punctuations to separate tokens.
        3. Transform different forms and misprints of expressions <<you>>, <<you are>> 
        to the unified form.
        4 Transform different forms and misprints of expression <<mother>> 
        to the unified form.
        5. Stemming.
        6. Replacing multiple letters.
        7. Transform bad words to the unified form using bad words dictionary [2].
        
    Learning process
    
        1. In train mode teacher sends comments and labels to agent which returns scores 
        (probabilities of each comment being insulting to another participant of the conversation).
        2. In test mode teacher sends comments to agent which returns scores 
        (probabilities of each comment being insulting to another participant of the conversation).
        
    Evaluating
        
        Target metric is AUC-ROC (Area Under ROC-curve).
        Teacher calculates chosen metrics over considered data.

    References:
    
        [1] - https://www.kaggle.com/c/detecting-insults-in-social-commentary/data
        
        [2] - https://aurbano.eu/blog/2008/04/04/bad-words-list/
        
"""