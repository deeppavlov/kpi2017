#!/usr/bin/env bash

mkdir -p ./build/ner

. ./env.sh

python3 ./utils/train_model.py -t deeppavlov.tasks.ner.agents \
                         -m deeppavlov.agents.ner.ner:NERAgent \
                         -mf ./build/ner \
                         --raw-dataset-path ./build/ner/ \
                         -dt train:ordered \
                         --learning_rate 0.01 \
                         --batchsize 2 \
                         --display-examples False \
                         --max-train-time -1 \
                         --validation-every-n-epochs 5 \
                         --log-every-n-epochs 1 \
                         --log-every-n-secs -1  \
                         --chosen-metric f1
