#!/usr/bin/env bash
# Training on GoPro (Sec. 4.1): 300 epochs, AdamW (lr 5e-4) with cosine annealing,
# 128x128 crops, and m / target timestamps sampled randomly for every input.

DATA_10DOWN=${DATA_10DOWN:-./datasets/GoPro_10down}
DATA_16DOWN=${DATA_16DOWN:-./datasets/GoPro_16down}

# GoPro-10x down: shutter period of 10 frames -> 20 target timestamps
python train.py --data_dir "$DATA_10DOWN" --num_frames 20 --exp_name afb_gopro_10down --batch_size 36 --multi_gpu

# GoPro-16x down: shutter period of 16 frames -> 32 target timestamps
python train.py --data_dir "$DATA_16DOWN" --num_frames 32 --exp_name afb_gopro_16down --batch_size 36 --multi_gpu
