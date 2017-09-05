#!/usr/bin/env bash

./run_paraphraser.sh | tail -n 1 | { read str; value=${str:11:20}; echo $value'>'0.8; } | bc -l
