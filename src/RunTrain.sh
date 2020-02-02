#!/usr/bin/env bash
# examples
# bash RunTrain.sh 0 cornell HRED 30 50
# bash RunTrain.sh 0 cornell VHRED 30 50

export CUDA_VISIBLE_DEVICES=$1

python train.py --data="$2" --model="$3" --batch_size="$4" --eval_batch_size="$4" --n_epoch="$5" --pretrained_wv=True --users=False

wait

