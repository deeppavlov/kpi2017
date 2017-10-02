#!/usr/bin/env bash

#parlai.agents.repeat_label.repeat_label
export CUDA_VISIBLE_DEVICES=0; python3 ./utils/train_model.py -t deeppavlov.tasks.coreference.agents:BaseTeacher \
                         -m deeppavlov.agents.coreference.agents:CoreferenceAgent \
                         -mf ./build/coreference \
                         --cor coreference \
                         --data-path ./build \
		         --language russian \
			 --name main \
			 --pretrained_model False \
                         --split 0.2 \
                         --random-seed None \
                         -dt train:ordered \
                         --batchsize 1 \
                         --display-examples True \
                         --max-train-time -1 \
                         --validation-every-n-epochs 10 \
                         --log-every-n-epochs 5 \
                         --log-every-n-secs -1 \
                         --chosen-metric conll-F-1 \

