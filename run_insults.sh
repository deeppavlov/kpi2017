#!/usr/bin/env bash

mkdir -p ./build/insults

. ./env.sh

python3 utils/train_model.py -t deeppavlov.tasks.insults.agents:FullTeacher \
                             -m deeppavlov.agents.insults.insults_agents:OneEpochAgent \
                             --model_file ./build/insults/log_reg \
                             -dt train:ordered \
                             --model_name log_reg \
                             --log-every-n-secs 10 \
                             --log-every-n-epochs 1 \
                             --validation-every-n-epochs 2 \
                             --raw-dataset-path ./build/insults/ \
                             --batchsize 64 \
                             --display-examples False \
                             --max-train-time -1 \
                             --num-epochs 1

python3 ./utils/train_model.py -t deeppavlov.tasks.insults.agents:FullTeacher \
                               -m deeppavlov.agents.insults.insults_agents:OneEpochAgent \
                               --model_file ./build/insults/svc \
                               -dt train:ordered \
                               --model_name svc \
                               --log-every-n-secs 10 \
                               --log-every-n-epochs 1 \
                               --validation-every-n-epochs 2 \
                               --raw-dataset-path ./build/insults/ \
                               --batchsize 64 \
                               --display-examples False \
                               --max-train-time -1 \
                               --num-epochs 1

python utils/train_model.py -t deeppavlov.tasks.insults.agents \
                            -m deeppavlov.agents.insults.insults_agents:InsultsAgent \
                            --model_file ./build/insults/cnn_word \
                            -dt train:ordered \
                            --model_name cnn_word  \
                            --log-every-n-secs 30 \
                            --raw-dataset-path ./build/insults/ \
                            --batchsize 64 \
                            --display-examples False \
                            --max-train-time -1 \
                            --num-epochs 50 \
                            --max_sequence_length 100 \
                            --learning_rate 0.01 \
                            --learning_decay 0.1 \
                            --filters_cnn 256 \
                            --embedding_dim 100 \
                            --kernel_sizes_cnn "3 3 3" \
                            --regul_coef_conv 0.001 \
                            --regul_coef_dense 0.001 \
                            --pool_sizes_cnn "2 2 2"  \
                            --dropout_rate 0.5 \
                            --dense_dim 100 \
                            --fasttext_model ./build/insults/reddit_fasttext_model.bin \
                            --fasttext_embeddings_dict ./build/insults/emb_dict.emb \
                            --cross-validation-splits-count 3


python utils/train_model.py -t deeppavlov.tasks.insults.agents \
                            -m deeppavlov.agents.insults.insults_agents:InsultsAgent \
                            --model_file ./build/insults/lstm_word \
                            -dt train:ordered \
                            --model_name lstm_word  \
                            --log-every-n-secs 30 \
                            --raw-dataset-path ./build/insults/ \
                            --batchsize 64 \
                            --display-examples False \
                            --max-train-time -1 \
                            --num-epochs 50 \
                            --max_sequence_length 100 \
                            --learning_rate 0.01 \
                            --learning_decay 0.1 \
                            --units_lstm 128 \
                            --embedding_dim 100 \
                            --regul_coef_lstm 0.001 \
                            --regul_coef_dense 0.001 \
                            --dropout_rate 0.5 \
                            --dense_dim 100 \
                            --fasttext_model ./build/insults/reddit_fasttext_model.bin \
                            --fasttext_embeddings_dict ./build/insults/emb_dict.emb \
                            --cross-validation-splits-count 3

python3 ./utils/train_model.py -t deeppavlov.tasks.insults.agents:FullTeacher \
                               -m deeppavlov.agents.insults.insults_agents:EnsembleInsultsAgent \
                               --model_file ./build/insults/insults_ensemble \
                               --model_files ./build/insults/cnn_word_0 \
                                             ./build/insults/cnn_word_1 \
                                             ./build/insults/cnn_word_2 \
                                             ./build/insults/lstm_word_0 \
                                             ./build/insults/lstm_word_1 \
                                             ./build/insults/lstm_word_2 \
                                             ./build/insults/log_reg \
                                             ./build/insults/svc \
                               --model_names cnn_word cnn_word cnn_word lstm_word lstm_word lstm_word log_reg svc \
                               --model_coefs 0.05 0.05 0.05 0.05 0.05 0.05 0.2 0.5 \
                               --datatype test \
                               --batchsize 64 \
                               --display-examples False \
                               --raw-dataset-path ./build/insults/ \
                               --max-train-time -1 \
                               --num-epochs 1 \
                               --max_sequence_length 100 \
                               --learning_rate 0.01 \
                               --learning_decay 0.1 \
                               --filters_cnn 256 \
                               --kernel_sizes_cnn "3 3 3" \
                               --regul_coef_conv 0.001 \
                               --regul_coef_dense 0.001 \
                               --pool_sizes_cnn "2 2 2"  \
                               --units_lstm 128 \
                               --embedding_dim 100 \
                               --regul_coef_lstm 0.001 \
                               --dropout_rate 0.5 \
                               --dense_dim 100 \
                               --fasttext_model ./build/insults/reddit_fasttext_model.bin

