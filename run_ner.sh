#!/usr/bin/env bash
python3 ./train_model.py -t parlai_tasks.ner.agents \
                         -m parlai_agents.repeat_label.repeat_label:RepeatLabelAgent \
                         -mf /tmp/paraphraser \
                         --batchsize 16 \
                         --raw-data-path /home/mikhail/Data/gareev \
                         --display-examples True \
                         --max-train-time -1 \
                         --datatype train
