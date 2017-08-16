#!/usr/bin/env bash
python3 ./train_model.py -t parlai_tasks.kaggle_insults.agents \
                         -m parlai_agents.repeat_label.repeat_label:RepeatLabelAgent \
                         -mf /tmp/insults \
                         --raw-dataset-path ~/Downloads/datasets \
                         --batchsize 1 \
                         --display-examples True \
                         --max-train-time 5
