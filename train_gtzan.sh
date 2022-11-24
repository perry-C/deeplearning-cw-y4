#!/usr/bin/env bash

#SBATCH --job-name=cw
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o sbatch_log/log_%j.out # STDOUT out
#SBATCH -e sbatch_log/log_%j.err # STDERR out
#SBATCH --gres=gpu:2
#SBATCH --time=0:20:00
#SBATCH --mem=20GB



# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

    python3 src/core/train_gtzan.py --reg 0 --aug 1
