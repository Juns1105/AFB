#!/usr/bin/env bash
# Evaluation on GoPro under the protocols of the paper (Sec. 4.1):
# symmetric exposure with fixed m, and RandEx (m = 0: randomly sampled per input).

DATA_10DOWN=${DATA_10DOWN:-./datasets/GoPro_10down}
DATA_16DOWN=${DATA_16DOWN:-./datasets/GoPro_16down}

# GoPro-10x down: shutter period of 10 frames, m in {1, 5, 9} and RandEx
for m in 1 5 9 0; do
    python test.py --data_dir "$DATA_10DOWN" --checkpoint checkpoints/afb_gopro_10down.pth \
        --num_frames 20 --left_m $m --right_m $m --save_dir results/gopro_10down_m$m
done

# GoPro-16x down: shutter period of 16 frames, m in {1, 5, 11, 15} and RandEx
for m in 1 5 11 15 0; do
    python test.py --data_dir "$DATA_16DOWN" --checkpoint checkpoints/afb_gopro_16down.pth \
        --num_frames 32 --left_m $m --right_m $m --save_dir results/gopro_16down_m$m
done
