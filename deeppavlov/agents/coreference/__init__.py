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
    agents.coreference

    This agent can be used to build coreference resolution model.
    Coreference resolution is the task of determining which mentions in a text refer to the same entity.
    This is end-to-end model[1], which can determine mentions in text independently, and also can used gold
    mentions. Gold mentions are used by default. You can select the desired mode. To do this,
    you need to set the True or False value in '--train_on_gold' parameter of train_coreference() function in 
    build.py module. Or, specify a parameter on the command line when starting the training:
    
        pyb train_coreference --train_on_gold True 
    
    The model works with data in the conll-2012 format. Model takes one conll file as input 
    and configure batch(-es) from it. In inference mode you can use conll file without the last column.
    Work with pure text in inference mode will be added later.
    Model predicts only mentions linking (clustering).

    Model architecture
        The model consists of two Bi-LSTM layers witch output is sent to span representation module, 
        that generate possible mentions spans. Spans representation from this module are sent to two
        fully-connected nets, that calculate mentions and antecedent scores. The hidden states from 
        this two nets are sent to fully-connected network that calculate the coreference scores. 
        To prevent over-fitting are used:
                    -- Dropout layers before each Dense layers.

        The optimizer is the Adam and Gradient Descent algorithm to choose from.

    Details
        Agent supposes that it receives train and dev set as observation.
        Model is trained for num-epochs number of epochs on train set.
        At the end of training model makes predictions on dev set.
        Agent can use only one document per step.
        
        Agent is to transformed conll file to dict:
        
            {
              "clusters": [[[1024,1024],[1024,1025]],[[876,876], [767,765], [541,544]]],
              "doc_key": "nw",
              "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
              "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
            }
        
        This dict is fed to the input of the model. To vectorize each token (word, signs) FastText embedding model
        is used [2]. Output embeddings are of 300 dimensions. Agent can use binary FastText model (model.bin),
        and already ready dict with embeddings (model.vec file). When the agent is initialized, the necessary
        embeddings .vec file and the chars dictionary are automatically loaded from [3].

        Model training process can be observed by using TensorBoard.

    References
        [1] https://homes.cs.washington.edu/~kentonl/pub/lhlz-emnlp.2017.pdf
        [2] https://github.com/facebookresearch/fastText
        [3] http://share.ipavlov.mipt.ru:8080/repository/embeddings/
"""
