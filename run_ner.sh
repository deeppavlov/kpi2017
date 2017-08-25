#!/usr/bin/env bash
python3 ./train_model.py -t parlai_tasks.ner.agents \
                         -m parlai_agents.ner.ner:NERAgent \
                         -mf /tmp/ner \
                         --learning_rate 0.001 \
                         --batchsize 16 \
                         --raw-data-path /home/mikhail/Data/gareev \
                         --display-examples True \
                         --max-train-time 20 \
                         --datatype train
