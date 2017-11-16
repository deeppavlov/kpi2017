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
    agents.coreference_scorer_model

    This agent can be used to build coreference resolution model.
    Coreference resolution is the task of determining which mentions in a text refer to the same entity.
    Gold mentions are used. It means that model uses extracted mention from a text.
    Model predicts only mentions linking (clustering).

    Model architecture
        Model consists of two parts: mention pair scorer and clustering over these scores.
        1. Mention pair scorer
            Scorer takes two mentions as input and outputs score in [0, 1].
            Scorer is a two hidden-layer FFNN with tanh activations and softmax on output with dropout 
            on inputs and all hidden layers.
            Optimizer: Adam
            
        2. Clustering
            Mention pair scorer generates scores for all mention pairs in text document and then
            agglomerative clustering from SciPy is used [1].
            
    Details
        Agent supposes that it receives train and dev set as observation.
        Mention pair scorer is trained for inner_epochs number of epochs on train set.
        At the end of training mention pair scorer makes predictions on dev set.
        Clustering threshold is adjusted to have highest CoNLL-F-1 score on dev set.
        
        Each mention is represented as pre-built features vector: word embeddings of words in mention, word embeddings
        of previous words, distance feature.
        Each mention pair is represented as concatenation of mention representations and mention pair features 
        (distance features).
        
        Model training process can be observed by using TensorBoard.
    
    References
        [1] https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.cluster.hierarchy.linkage.html
"""
