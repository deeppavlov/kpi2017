#!/usr/bin/env bash

mkdir -p ./build

python utils/train_model.py -t deeppavlov.tasks.insults.agents \
                            -m deeppavlov.agents.insults.insults_agents:InsultsAgent \
                            --model_file ./build/cnn_word \
                            -dt train:ordered \
                            --model_name cnn_word  \
                            --log-every-n-secs 60 \
                            --raw-dataset-path ./build/ \
                            --batchsize 64 \
                            --display-examples False \
                            --max-train-time -1 \
                            --num-epochs 1000 \
                            --max_sequence_length 100 \
                            --learning_rate 0.01 \
                            --learning_decay 0.1 \
                            --filters_cnn 256 \
                            --embedding_dim 100 \
                            --kernel_sizes_cnn "1 2 3" \
                            --regul_coef_conv 0.001 \
                            --regul_coef_dense 0.01 \
                            --dropout_rate 0.5 \
                            --dense_dim 100 \
                            --fasttext_model ./build/reddit_fasttext_model.bin \
                            --fasttext_embeddings_dict ./build/emb_dict.emb \
                            --cross-validation-splits-count 3 \
                            -ve 10 \
                            -vp 5 \
                            --chosen-metric auc

python3 ./utils/train_model.py -t deeppavlov.tasks.insults.agents:FullTeacher \
                               -m deeppavlov.agents.insults.insults_agents:EnsembleInsultsAgent \
                               --model_file ./build/insults_ensemble \
                               --model_files ./build/cnn_word_0 \
                                             ./build/cnn_word_1 \
                                             ./build/cnn_word_2 \
                               --model_names cnn_word cnn_word cnn_word \
                               --model_coefs 0.3333333 0.3333333 0.3333334 \
                               --datatype test \
                               --batchsize 64 \
                               --display-examples False \
                               --raw-dataset-path ./build/ \
                               --max_sequence_length 100 \
                               --filters_cnn 256 \
                               --kernel_sizes_cnn "1 2 3" \
                               --embedding_dim 100 \
                               --dense_dim 100 \
                               --fasttext_model ./build/reddit_fasttext_model.bin

