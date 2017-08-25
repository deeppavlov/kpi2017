#!/usr/bin/env bash
python3 ./utils/train_model.py -t deeppavlov.tasks.ner.agents \
                         -m deeppavlov.agents.ner.ner:NERAgent \
                         -mf /tmp/ner \
                         --learning_rate 0.001 \
                         --batchsize 16 \
                         --raw-data-path /home/mikhail/Data/gareev \
                         --display-examples True \
                         --max-train-time 20 \
                         --datatype train
