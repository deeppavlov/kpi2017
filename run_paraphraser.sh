#!/usr/bin/env bash
python3 ./train_model.py -t parlai_tasks.paraphrases.agents \
                         -m parlai_agents.repeat_label.repeat_label:RepeatLabelAgent \
                         -mf /tmp/paraphraser \
                         --display-examples True \
                         --max-train-time 10