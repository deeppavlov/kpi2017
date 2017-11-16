"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
    agents.insults
    
    Problem statement

        The problem being solved is to detect whether a comment from a conversation online 
        would be considered as an insult to another participant of the conversation.
        
        This task is considered as a binary classification problem 
        where <<1>> denotes comment insulting to another participant of the conversation, 
        and <<0>> - comment non-insulting to another participant of the conversation.
        
        There are presented four possible models.
        Preprocessed comments are given to all of them.
    
    Models architecture
    
        I.  Sklearn models
    
            To vectorize text data TFIDF-vectorizer from sklearn is used.
            Using SelectKBest (Chi-square statistics) feature selection is being applied.
            Then two models could be trained over the sample-feature matrix:    
        
            1. Logistic regression
            2. Support Vector Classifier
        
        II. Neural models
            
            To vectorize each token (word, signs) of comment FastText embedding model [1] 
            over a set of comments from Reddit web-site [2].
            Output embeddings are of 100 dimensions.
            Then two models could be trained over embedded 
            padded up to particular number of tokens comments:
            
            1. Shallow-and-wide Convolutional Neural Network [3].
            
                The NN consists of three separate 1D-convolutions with given (better different) 
                kernel sizes each of which is followed by global max pooling layer.
                The outputs are concatenated and sent to two consistent dense layers.
                To prevent over-fitting are used:
                    -- L2-regularization of weights in Conv1D and Dense layers, 
                    -- batch normalization [4],
                    -- Dropout layers before each Dense layers.
                All layers except last Dense use ReLU activation, last one uses sigmoid.
                Loss function is bnary cross-entropy.
                Optimizer is Adam algorithm.
                            
            2. Bidirectional LSTM.
            
                The NN consists of BiLSTM layer which output is sent to two consistent dense layers.
                To prevent over-fitting are used:
                    -- L2-regularization of weights in Conv1D and Dense layers, 
                    -- batch normalization [4],
                    -- Dropout layers before each Dense layers.
                BiLSTM layer uses tanh activation, the first Dense - ReLU activation,
                the last Dense layer - sigmoid activation.   
                Loss function is bnary cross-entropy.
                Optimizer is Adam algorithm.  

    References:
    
        [1] - Enriching Word Vectors with Subword Information / Piotr Bojanowski, Edouard Grave, Ar-
    mand Joulin, Tomas Mikolov // arXiv preprint arXiv:1607.04606. — 2016.
    
        [2] - http://files.pushshift.io/reddit/comments/
    
        [3] - Le, Hoa T. Do Convolutional Networks need to be Deep for Text Classification? / Hoa T Le,
    Christophe Cerisara, Alexandre Denis // arXiv preprint arXiv:1707.04108. — 2017.

        [4] - Ioffe, Sergey. Batch normalization: Accelerating deep network training by reducing internal co-
variate shift / Sergey Ioffe, Christian Szegedy // International Conference on Machine Learning.
— 2015. — Pp. 448–456.

"""
