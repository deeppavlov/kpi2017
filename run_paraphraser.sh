#!/usr/bin/env bash
python3 ./train_model.py -t parlai_tasks.paraphrases.agents \
                         -m parlai_agents.paraphraser.paraphraser:ParaphraserAgent \
                         -mf /tmp/paraphraser \
                         --batchsize 256 \
                         --display-examples False \
                         --max-train-time -1 \
                         --num-epochs 10 \
                         --log-every-n-secs -1 \
                         --log-every-n-epochs 1 \
                         --learning_rate 0.00001 \
                         --hidden_dim 200 \
                         --embedding_file 'yalen_sg_word_vectors_300.txt' \
                         --validation-every-n-epochs 1
#                         --pretrained_model '/tmp/my_model/my_model'
#                         --validation-patience 5 \
