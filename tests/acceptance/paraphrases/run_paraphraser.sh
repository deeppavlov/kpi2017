#!/usr/bin/env bash
python3 ./utils/train_model.py -t deeppavlov.tasks.paraphrases.agents \
                         -m deeppavlov.agents.paraphraser.paraphraser:EnsembleParaphraserAgent \
                         -mf /tmp/paraphraser_maxpool_match \
                         --model_files /tmp/maxpool_match \
                         --datatype test \
                         --batchsize 256 \
                         --display-examples False \
                         --fasttext_embeddings_dict "/tmp/paraphraser.emb" \
                         --fasttext_model '/tmp/ft_0.8.3_nltk_yalen_sg_300.bin' \
                         --cross-validation-splits-count 5 \
                         --chosen-metric f1
