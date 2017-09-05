#!/usr/bin/env bash
python3 ./utils/train_model.py -t deeppavlov.tasks.ner.agents \
                         -m deeppavlov.agents.ner.ner:NERAgent \
                         -mf /tmp/ner \
                         -dt train:ordered \
                         --learning_rate 0.01 \
                         --batchsize 16 \
                         --raw-data-path /home/mikhail/Data/gareev \
                         --display-examples True \
                         --max-train-time -1 \
                         --validation-every-n-secs 10 \
                         --chosen-metric f1 \
                         --datatype train
