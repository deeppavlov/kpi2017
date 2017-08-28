#!/usr/bin/env bash
python3 utils/train_model.py -t squad \
                         -m deeppavlov.agents.squad.squad:SquadAgent \
                         -mf /tmp/squad_model \
                         --batchsize 5 \
                         --display-examples False \
                         --max-train-time -1 \
                         --num-epochs -1 \
                         --log-every-n-secs 20 \
                         --log-every-n-epochs -1 \
                         --validation-every-n-secs 600 \
                         --chosen-metric f1 \
                         --embedding_file '/home/anatoly/data/fasttext/wiki-news-300d-1M.vec'
