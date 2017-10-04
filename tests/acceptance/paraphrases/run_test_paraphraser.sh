#!/usr/bin/env bash
result=`python3 ./utils/train_model.py -t deeppavlov.tasks.paraphrases.agents \
                         -m deeppavlov.agents.paraphraser.paraphraser:EnsembleParaphraserAgent \
                         -mf ./build/paraphraser_maxpool_match \
                         --model_files ./build/maxpool_match \
                         --datatype test \
                         --batchsize 256 \
                         --display-examples False \
                         --fasttext_embeddings_dict "./build/paraphraser.emb" \
                         --fasttext_model './build/ft_0.8.3_nltk_yalen_sg_300.bin' \
                         --cross-validation-splits-count 5 \
                         --chosen-metric f1 | tail -n 1 | { read str; value=${str:11:20}; echo $value'>'0.8; } | bc -l`
if [[ result==0 ]]; then
	exit 1
else
	exit 0
fi
