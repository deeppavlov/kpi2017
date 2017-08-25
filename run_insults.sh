#!/usr/bin/env bash

python3 utils/train_model.py -t deeppavlov.tasks.insults.agents \
                         -m deeppavlov.agents.insults.insults_agents:InsultsAgent \
                         -mf C:/Users/Dilyara/ParlAI/tmp/insults_cnn_word \
                         -dt train:ordered \
                         --model_name cnn_word \
                         --log-every-n-secs 5 \
                         --raw-dataset-path C:/Users/Dilyara/Documents/DataScience/Insults_kaggle/data \
                         --batchsize 64 \
                         --display-examples True \
                         --max-train-time 100 \
                         --num-epochs 5 \
                         --max_sequence_length 200 \
                         --learning_rate 0.1 \
                         --learning_decay 0.1 \
                         --num_filters 64 \
                         --kernel_sizes "3 4 5" \
                         --regul_coef_conv 0.01 \
                         --regul_coef_dense 0.01 \
                         --pool_sizes "2 2 2" \
                         --dropout_rate 0.5 \
                         --dense_dim 100

python3 utils/train_model.py -t deeppavlov.tasks.insults.agents \
                         -m deeppavlov.agents.insults.insults_agents:OneEpochAgent \
                         -mf C:/Users/Dilyara/ParlAI/tmp/insults_log_reg \
                         -dt train:ordered \
                         --model_name log_reg \
                         --log-every-n-secs 5 \
                         --raw-dataset-path C:/Users/Dilyara/Documents/DataScience/Insults_kaggle/data \
                         --batchsize 64 \
                         --display-examples True \
                         --max-train-time 100 \
                         --num-epochs 1

python3 ./train_model.py -t deeppavlov.tasks.insults.agents \
                         -m deeppavlov.agents.insults.insults_agents:OneEpochAgent \
                         -mf C:/Users/Dilyara/ParlAI/tmp/insults_svc \
                         -dt train:ordered \
                         --model_name svc \
                         --log-every-n-secs 5 \
                         --raw-dataset-path C:/Users/Dilyara/Documents/DataScience/Insults_kaggle/data \
                         --batchsize 64 \
                         --display-examples True \
                         --max-train-time 100 \
                         --num-epochs 1

