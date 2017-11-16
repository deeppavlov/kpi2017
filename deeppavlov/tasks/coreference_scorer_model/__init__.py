# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    tasks.coreference_scorer_model
    
    This teacher prepares data from RuCor corpus [1][2].
     
    Data preprocessing
        1. RuCor corpus files are converted to standard conll format [3]. 
        2. RuCor doesn't provide train/dev/test data splitting, teacher makes random splitting. 
    
    Learning process
        1. In train mode teacher acts by sending whole train and dev set as conll files.
        2. In valid/test mode teacher acts by sending test set.
    
    Evaluating
        For evaluation official CoNLL-2012 [3] scorer script is used.
        Target score: CoNLL-F-1.

    References
        [1] RuCor - Russian coreference corpus http://rucoref.maimbava.net/
        [2] Evaluating Anaphora and Coreference Resolution for Russian / S. Toldova, A. Roytberg, A. A. Ladygina et al. //
            Komp’juternaja lingvistika i intellektual’nye tehnologii. Po materialam ezhegodnoj Mezhdunarodnoj konferencii
            Dialog. — 2014. — Pp. 681–695.
        [3] CoNLL-2012 shared task http://conll.cemantix.org/2012/
"""
