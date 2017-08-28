#!/usr/bin/env bash
python3 ./train_model.py -t squad \
                         -m parlai_agents.squad.squad:SquadAgent \
                         -mf /tmp/squad_model \
                         --batchsize 64 \
                         --display-examples True \
                         --max-train-time 10 \
                         --num-epochs -1 \
                         --log-every-n-secs 2 \
                         --log-every-n-epochs -1 \
                         --validation-every-n-epochs -1 \
                         --embedding_file '/home/anatoly/data/fasttext/wiki-news-300d-1M.vec'
