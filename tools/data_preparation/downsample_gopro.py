"""Step 1: downsample the 240 fps GoPro frames to 320x180.

Input:  <input_dir>/<split>/<video>/*.png                (GOPRO_Large_all)
Output: <output_dir>/<split>/<video>/{imgs/*.png, fps.txt}   (input layout of the rpg_vid2e upsampler)
"""
import argparse
import glob
import os

import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', required=True, help='GOPRO_Large_all root')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--fps', type=int, default=240)
    args = parser.parse_args()

    for split in args.splits:
        split_dir = os.path.join(args.input_dir, split)
        videos = sorted(v for v in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, v)))
        for video in videos:
            out_dir = os.path.join(args.output_dir, split, video)
            os.makedirs(os.path.join(out_dir, 'imgs'), exist_ok=True)
            for path in tqdm(sorted(glob.glob(os.path.join(split_dir, video, '*.png'))), desc=f'{split}/{video}'):
                img = cv2.resize(cv2.imread(path), (args.width, args.height), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(out_dir, 'imgs', os.path.basename(path)), img)
            with open(os.path.join(out_dir, 'fps.txt'), 'w') as f:
                f.write(f'{args.fps}\n')


if __name__ == '__main__':
    main()
