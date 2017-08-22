#!/usr/bin/env bash
python3 ./train_model.py -t parlai_tasks.paraphrases.agents \
                         -m parlai_agents.paraphraser.paraphraser:ParaphraserAgent \
                         -mf /tmp/paraphraser \
                         --datatype train:ordered \
                         --batchsize 256 \
                         --display-examples False \
                         --max-train-time -1 \
                         --num-epochs -1 \
                         --log-every-n-secs -1 \
                         --log-every-n-epochs 1 \
                         --learning_rate 0.0001 \
                         --hidden_dim 200 \
                         --validation-every-n-epochs 5 \
                         --fasttext_dir '~/github/fastText' \
                         --fasttext_model '/tmp/model_yalen_sg_300.bin'
#                         --pretrained_model '/tmp/paraphraser'
#                         --validation-patience 5 \
