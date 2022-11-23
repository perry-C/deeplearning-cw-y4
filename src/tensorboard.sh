#!/usr/bin/env bash


module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch 

PORT=$((($UID-6025) % 65274))

echo $PORT
hostname -s

tensorboard --logdir logs --port $PORT --bind_all