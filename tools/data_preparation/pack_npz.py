"""Step 4: pack consecutive shutter periods into the .npz files read by data/gopro.py.

A shutter period spans `period` frames of the 240 fps video (10 for GoPro-10x down, 16 for GoPro-16x down).
Its events are those simulated between its first frame and the first frame of the next period.

Input:
    <frames_dir>/<split>/<video>/imgs/*.png           320x180 frames (step 1)
    <upsampled_dir>/<split>/<video>/timestamps.txt    timestamps of the upsampled frames (step 2)
    <events_dir>/<split>/<video>/*.npz                simulated events (step 3)
Output:
    <output_dir>/<split>/<video>/<xxxxxx_yyyyyy>.npz  two consecutive shutter periods
"""
import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def frame_boundaries(timestamps_file, fps=240, tolerance_ns=3000):
    """Indices of the upsampled timestamps that coincide with frames of the original video."""
    timestamps_ns = (np.genfromtxt(timestamps_file, dtype='float64') * 1e9).astype('int64')
    frame_interval_ns = int(1 / fps * 1e9)
    return [i for i, t in enumerate(timestamps_ns) if t % frame_interval_ns < tolerance_ns]


def load_events(event_files, start, end):
    events = [np.load(event_files[i]) for i in range(start, end)]
    return {k: np.concatenate([e[k] for e in events]) for k in ('x', 'y', 't', 'p')}


def pack_video(frames_dir, timestamps_file, events_dir, out_dir, period):
    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    event_files = sorted(glob.glob(os.path.join(events_dir, '*.npz')))
    boundaries = frame_boundaries(timestamps_file)

    def boundary(frame_index):
        # The last period ends at the last timestamp.
        return boundaries[min(frame_index, len(boundaries) - 1)]

    def load_period(k):
        frames = {j: cv2.imread(f) for j, f in enumerate(frame_files[k * period:(k + 1) * period])}
        events = load_events(event_files, boundary(k * period), boundary((k + 1) * period))
        return frames, events

    os.makedirs(out_dir, exist_ok=True)
    num_periods = len(frame_files) // period
    left = load_period(0)
    for k in tqdm(range(num_periods - 1), desc=os.path.basename(out_dir)):
        right = load_period(k + 1)
        name = f'{k * period:06d}_{(k + 1) * period:06d}.npz'
        np.savez(os.path.join(out_dir, name),
                 left_frames=left[0], right_frames=right[0], left_event=left[1], right_event=right[1])
        left = right


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames_dir', required=True)
    parser.add_argument('--upsampled_dir', required=True)
    parser.add_argument('--events_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--period', type=int, required=True, help='frames per shutter period: 10 or 16')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    args = parser.parse_args()

    for split in args.splits:
        for video in sorted(os.listdir(os.path.join(args.events_dir, split))):
            pack_video(frames_dir=os.path.join(args.frames_dir, split, video, 'imgs'),
                       timestamps_file=os.path.join(args.upsampled_dir, split, video, 'timestamps.txt'),
                       events_dir=os.path.join(args.events_dir, split, video),
                       out_dir=os.path.join(args.output_dir, split, video),
                       period=args.period)


if __name__ == '__main__':
    main()
