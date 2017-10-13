#!/bin/bash

source activate py3
export CUDA_VISIBLE_DEVICES=3; KERAS_BACKEND=tensorflow; pyb train_insults