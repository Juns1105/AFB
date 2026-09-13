"""Step 3: simulate events from the upsampled sequences with ESIM.

Requires esim_torch from rpg_vid2e: https://github.com/uzh-rpg/rpg_vid2e

Input:  <input_dir>/<split>/<video>/{imgs/*.png, timestamps.txt}   (output of rpg_vid2e/upsampling/upsample.py)
Output: <output_dir>/<split>/<video>/%010d.npz, the events between each pair of consecutive upsampled frames

The positive and negative contrast thresholds of each video are sampled from N(mu, sigma^2).
"""
import argparse
import glob
import os
import random

import cv2
import esim_torch
import numpy as np
import torch
from tqdm import tqdm


def simulate_video(video_dir, out_dir, contrast_threshold_neg, contrast_threshold_pos, refractory_period_ns):
    esim = esim_torch.ESIM(contrast_threshold_neg, contrast_threshold_pos, refractory_period_ns)
    timestamps = np.genfromtxt(os.path.join(video_dir, 'timestamps.txt'), dtype='float64')
    timestamps_ns = torch.from_numpy((timestamps * 1e9).astype('int64')).cuda()
    image_files = sorted(glob.glob(os.path.join(video_dir, 'imgs', '*.png')))

    os.makedirs(out_dir, exist_ok=True)
    counter = 0
    for image_file, timestamp_ns in zip(tqdm(image_files, desc=os.path.basename(video_dir)), timestamps_ns):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        log_image = torch.from_numpy(np.log(image.astype('float32') / 255 + 1e-5)).cuda()
        events = esim.forward(log_image, timestamp_ns)
        if events is None:  # the first image produces no events
            continue
        np.savez(os.path.join(out_dir, '%010d.npz' % counter), **{k: v.cpu().numpy() for k, v in events.items()})
        counter += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', required=True, help='root of the upsampled sequences')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument('--mu', type=float, default=0.2, help='mean of the contrast thresholds')
    parser.add_argument('--sigma', type=float, default=0.03, help='standard deviation of the contrast thresholds')
    parser.add_argument('--refractory_period_ns', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None, help='seed for sampling the contrast thresholds')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    for split in args.splits:
        split_dir = os.path.join(args.input_dir, split)
        for video in sorted(os.listdir(split_dir)):
            video_dir = os.path.join(split_dir, video)
            if not os.path.isfile(os.path.join(video_dir, 'timestamps.txt')):
                continue
            contrast_threshold_neg = random.gauss(args.mu, args.sigma)
            contrast_threshold_pos = random.gauss(args.mu, args.sigma)
            print(f'{split}/{video}: cn={contrast_threshold_neg:.4f}, cp={contrast_threshold_pos:.4f}')
            simulate_video(video_dir, os.path.join(args.output_dir, split, video),
                           contrast_threshold_neg, contrast_threshold_pos, args.refractory_period_ns)


if __name__ == '__main__':
    main()
