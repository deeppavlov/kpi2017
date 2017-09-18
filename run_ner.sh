#!/usr/bin/env bash
python3 ./utils/train_model.py -t deeppavlov.tasks.ner.agents \
                         -m deeppavlov.agents.ner.ner:NERAgent \
                         -mf /tmp/ner \
                         -dt train:ordered \
                         --learning_rate 0.01 \
                         --batchsize 2 \
                         --raw-data-path /home/mikhail/Data/gareev \
                         --display-examples False \
                         --max-train-time -1 \
                         --validation-every-n-epochs 5 \
                         --log-every-n-epochs 1 \
                         --log-every-n-secs -1  \
                         --chosen-metric f1
