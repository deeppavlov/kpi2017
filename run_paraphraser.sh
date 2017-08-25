#!/usr/bin/env bash
python3 ./utils/train_model.py -t deeppavlov.tasks.paraphrases.agents \
                         -m deeppavlov.agents.paraphraser.paraphraser:ParaphraserAgent \
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
                         --fasttext_embeddings_dict "/tmp/paraphraser.emb" \
                         --fasttext_model '/tmp/ft_0.8.3_nltk_yalen_sg_300.bin' \
                         --cross-validation-seed 50 \
                         --cross-validation-splits-count 5 \
                         --validation-patience 3
#                         --pretrained_model '/tmp/paraphraser'

python3 ./utils/train_model.py -t deeppavlov.tasks.paraphrases.agents \
                         -m deeppavlov.agents.paraphraser.paraphraser:EnsembleParaphraserAgent \
                         -mf /tmp/paraphraser \
                         --model_files /tmp/paraphraser \
                         --datatype test \
                         --batchsize 256 \
                         --display-examples False \
                         --fasttext_embeddings_dict "/tmp/paraphraser.emb" \
                         --fasttext_model '/tmp/ft_0.8.3_nltk_yalen_sg_300.bin' \
                         --cross-validation-splits-count 5