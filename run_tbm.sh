#!/usr/bin/env bash
export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
python3 utils/train_model.py -t deeppavlov.tasks.ud_pos_english.agents \
                       -m deeppavlov.agents.tbm_pos.tbm:NaiveAgent \
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