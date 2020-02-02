#!/usr/bin/env bash
# examples
# bash RunExportTestSamples.sh 0 cornell HRED 30 3 1 30.pkl
# bash RunExportTestSamples.sh 0 cornell VHRED 30 3 1 30.pkl

export CUDA_VISIBLE_DEVICES=$1

python export_test_responses.py --data="$2" --model="$3" --batch_size="$4" --pretrained_wv=True --users=False --n_context="$5" --n_sample_step="$6" --checkpoint="$7"

wait

