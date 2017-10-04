#!/usr/bin/env bash

mkdir -p ./build/paraphraser/

. ./env.sh

python3 ./utils/train_model.py -t deeppavlov.tasks.paraphrases.agents \
                         -m deeppavlov.agents.paraphraser.paraphraser:ParaphraserAgent \
                         -mf ./build/paraphraser/paraphraser \
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
                         --fasttext_embeddings_dict "./build/paraphraser/paraphraser.emb" \
                         --fasttext_model './build/paraphraser/ft_0.8.3_nltk_yalen_sg_300.bin' \
                         --cross-validation-seed 50 \
                         --cross-validation-splits-count 5 \
                         --validation-patience 3 \
                         --chosen-metric f1
#                         --pretrained_model './build/paraphraser/paraphraser'

python3 ./utils/train_model.py -t deeppavlov.tasks.paraphrases.agents \
                         -m deeppavlov.agents.paraphraser.paraphraser:EnsembleParaphraserAgent \
                         -mf ./build/paraphraser/paraphraser \
                         --model_files ./build/paraphraser/paraphraser \
                         --datatype test \
                         --batchsize 256 \
                         --display-examples False \
                         --fasttext_embeddings_dict "./build/paraphraser/paraphraser.emb" \
                         --fasttext_model './build/paraphraser/ft_0.8.3_nltk_yalen_sg_300.bin' \
                         --cross-validation-splits-count 5 \
                         --chosen-metric f1
