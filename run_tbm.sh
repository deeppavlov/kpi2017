#!/usr/bin/env bash
python3 train_model.py -t parlai_tasks.ud_pos_english.agents \
                       -m parlai_agents.tbm_pos.tbm:NaiveAgent \
                       -dt train \
                       -mf /tmp/model_tbm \
                       --max-train-time 600 \
                       --validation-every-n-secs -1 \
                       --log-every-n-secs 2 \
                       --display-examples False \
                       --cuda True \
                       --batchsize 64 \
                       --learning_rate 0.01 \
                       --beam_size 8 \
                       --trainer_type naive
