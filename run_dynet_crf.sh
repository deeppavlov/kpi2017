#!/usr/bin/env bash
export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
python3 utils/train_model.py -t deeppavlov.tasks.ud_pos_english.agents \
                       -m deeppavlov.agents.dynet_pos.tbm:NaiveAgent \
                       -dt train \
                       -mf /tmp/model_tbm \
                       --max-train-time -1 \
                       --validation-every-n-secs 60 \
                       --log-every-n-secs 10 \
                       --display-examples False \
                       --batchsize 5 \
                       --learning_rate 0.05 \
                       --depth 7 \
                       --hidden-size 51 \
                       --word-dim 5000
