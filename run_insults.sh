#!/usr/bin/env bash

PATH_TO_PARLAI="C:/Users/Dilyara/ParlAI"
PATH_TO_RAW_DATA="C:/Users/Dilyara/Documents/DataScience/Insults_kaggle/data"

python3 utils/train_model.py -t deeppavlov.tasks.insults.agents:FullTeacher \
                             -m deeppavlov.agents.insults.insults_agents:OneEpochAgent \
                             --model_file "$PATH_TO_PARLAI"/tmp/insults_log_reg/log_reg \
                             -dt train:ordered \
                             --model_name log_reg \
                             --log-every-n-secs 10 \
                             --log-every-n-epochs 1 \
                             --validation-every-n-epochs 2 \
                             --raw-dataset-path "$PATH_TO_RAW_DATA" \
                             --batchsize 64 \
                             --display-examples False \
                             --max-train-time -1 \
                             --num-epochs 1

python3 ./utils/train_model.py -t deeppavlov.tasks.insults.agents:FullTeacher \
                               -m deeppavlov.agents.insults.insults_agents:OneEpochAgent \
                               --model_file "$PATH_TO_PARLAI"/tmp/insults_svc/svc \
                               -dt train:ordered \
                               --model_name svc \
                               --log-every-n-secs 10 \
                               --log-every-n-epochs 1 \
                               --validation-every-n-epochs 2 \
                               --raw-dataset-path "$PATH_TO_RAW_DATA" \
                               --batchsize 64 \
                               --display-examples False \
                               --max-train-time -1 \
                               --num-epochs 1

python utils/train_model.py -t deeppavlov.tasks.insults.agents \
                            -m deeppavlov.agents.insults.insults_agents:InsultsAgent \
                            --model_file "$PATH_TO_PARLAI"/tmp/insults_cnn_word/cnn_word \
                            -dt train:ordered \
                            --model_name cnn_word  \
                            --log-every-n-secs 30 \
                            --raw-dataset-path "$PATH_TO_RAW_DATA" \
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
                            --fasttext_model "$PATH_TO_RAW_DATA"/reddit_fasttext_model.bin \
                            --fasttext_embeddings_dict "$PATH_TO_RAW_DATA"/emb_dict.emb \
                            --cross-validation-splits-count 3


python utils/train_model.py -t deeppavlov.tasks.insults.agents \
                            -m deeppavlov.agents.insults.insults_agents:InsultsAgent \
                            --model_file "$PATH_TO_PARLAI"/tmp/insults_lstm_word/lstm_word \
                            -dt train:ordered \
                            --model_name lstm_word  \
                            --log-every-n-secs 30 \
                            --raw-dataset-path "$PATH_TO_RAW_DATA" \
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
                            --fasttext_model "$PATH_TO_RAW_DATA"/reddit_fasttext_model.bin \
                            --fasttext_embeddings_dict "$PATH_TO_RAW_DATA"/emb_dict.emb \
                            --cross-validation-splits-count 3

python3 ./utils/train_model.py -t deeppavlov.tasks.insults.agents:FullTeacher \
                               -m deeppavlov.agents.insults.insults_agents:EnsembleInsultsAgent \
                               --model_file "$PATH_TO_PARLAI"//tmp/insults_ensemble \
                               --model_files "$PATH_TO_PARLAI"/tmp/insults_cnn_word/cnn_word_0 \
                                             "$PATH_TO_PARLAI"/tmp/insults_cnn_word/cnn_word_1 \
                                             "$PATH_TO_PARLAI"/tmp/insults_cnn_word/cnn_word_2 \
                                             "$PATH_TO_PARLAI"/tmp/insults_lstm_word/lstm_word_0 \
                                             "$PATH_TO_PARLAI"/tmp/insults_lstm_word/lstm_word_1 \
                                             "$PATH_TO_PARLAI"/tmp/insults_lstm_word/lstm_word_2 \
                                             "$PATH_TO_PARLAI"/tmp/insults_log_reg/log_reg \
                                             "$PATH_TO_PARLAI"/tmp/insults_svc/svc \
                               --model_names cnn_word cnn_word cnn_word lstm_word lstm_word lstm_word log_reg svc \
                               --model_coefs 0.05 0.05 0.05 0.05 0.05 0.05 0.2 0.5 \
                               --datatype test \
                               --batchsize 64 \
                               --display-examples False \
                               --raw-dataset-path "$PATH_TO_RAW_DATA" \
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
                               --fasttext_model "$PATH_TO_RAW_DATA"/reddit_fasttext_model.bin

