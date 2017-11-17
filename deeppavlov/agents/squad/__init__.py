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
    agents.squad

    Extractive model for solving question answering problem on SQuAD dataset. Given question and supporting
    paragraph of text, model produces answer in the form of two pointers to the start and the end of answer
    in the supporting paragraph.

    Model architecture is based of FastQA[1] and DRQA[2]. Roughly the architecture consists of the following blocks:
        1. bi-LSTM encoders for question and text passage (3 layers, each layer output is concatenated to one output repr.)
        2. attention over question to produce one fused question representation (instead of sequence of token representation)
        3. answer start and end pointers (two layer dense with dropout)

    Model use pretrained glove embeddings glove.840B.300d.txt [6], and incorporate some additional features for each word
    such as "word in question", tf (term frequency) etc..

    Training the model:
        pyb train_squad

    Evaluating the model:
        pyb run_unit_tests -P unittest_test_method_prefix="test_squad"

    References
        [1] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang:
            SQuAD: 100,000+ Questions for Machine Comprehension of Text (2016)
        [2] Dirk Weissenborn, Georg Wiese, Laura Seiffe:
            FastQA: A Simple and Efficient Neural Architecture for Question Answering (2017)
        [3] Chen, Danqi and Fisch, Adam and Weston, Jason and Bordes, Antoine:
            Reading Wikipedia to Answer Open-Domain Questions (2017)
        [4] Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi:
            Bidirectional Attention Flow for Machine Comprehension (2017)
        [5] Minghao Hu, Yuxing Peng, Xipeng Qiu:
            Reinforced Mnemonic Reader for Machine Comprehension (2017)
        [6] Jeffrey Pennington, Richard Socher, Christopher D. Manning:
            Global Vectors for Word Representation (2014)

"""