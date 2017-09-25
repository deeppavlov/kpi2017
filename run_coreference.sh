#!/usr/bin/env bash

mkdir -p ./build/coreference

# Build custom kernels.
#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

# Linux (pip)
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0

# Linux (build from source)
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I $TF_INC -fPIC

# Mac
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0  -undefined dynamic_lookup

python3 ./utils/train_model.py -t deeppavlov.tasks.coreference.agents:BaseTeacher \
                         -m parlai.agents.repeat_label.repeat_label:RepeatLabelAgent \
                         -mf ./build/coreference \
                         --cor coreference \
                         --datapath ./build \
		         --language russian \
                         --split 0.2 \
                         --random_seed None \
                         -dt train:ordered \
                         --learning_rate 0.01 \
                         --batchsize 1 \
                         --display-examples True \
                         --max-train-time -1 \
                         --validation-every-n-epochs 1 \
                         --log-every-n-epochs 1 \
                         --log-every-n-secs -1  
