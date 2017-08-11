#!/usr/bin/env bash
python3 ./train_model.py -t parlai_tasks.paraphrases.agents \
                         -m parlai_agents.paraphraser.paraphraser:ParaphraserAgent \
                         -mf /tmp/paraphraser \
                         --batchsize 256 \
                         --display-examples False \
                         --max-train-time -1 \
                         --num-epochs 200 \
                         --log-every-n-secs -1 \
                         --log-every-n-epochs 1 \
                         --model_file '/tmp/my_model'\
                         --learning_rate 0.0001 \
                         --hidden_dim 50
#                         --embedding_file 'yalen_sg_word_vectors_300.txt'\