import glob
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .event_utils import events_to_stack


class GoProTestDataset(Dataset):
    """GoPro sequences packed as .npz files for exposure-agnostic VFI evaluation.

    Expected layout: <data_dir>/<split>/<video>/<xxxxxx_yyyyyy>.npz, where each file holds
    two consecutive shutter periods:
        left_frames, right_frames: dict {i: (H, W, 3) uint8 BGR sharp frame}
        left_event, right_event:   dict {'x', 'y', 't', 'p'} event arrays

    The blurry input of each shutter period is synthesized by averaging its first m sharp
    frames. m = 0 samples m randomly within the shutter period (RandEx).
    """

    def __init__(self, data_dir, num_bins, left_m=0, right_m=0, split='test', video_name=None):
        self.num_bins = num_bins
        self.left_m = left_m
        self.right_m = right_m

        split_dir = os.path.join(data_dir, split)
        videos = [video_name] if video_name is not None else os.listdir(split_dir)
        self.data_list = []
        for video in videos:
            self.data_list.extend(sorted(glob.glob(os.path.join(split_dir, video, '*.npz'))))

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def synthesize_blur(frames, m):
        h, w, c = frames[0].shape
        blurry = np.zeros((h, w, c), dtype=np.float32)
        for i in range(m):
            blurry += frames[i]
        blurry /= m
        blurry = cv2.cvtColor(blurry, cv2.COLOR_BGR2RGB).astype(np.uint8)
        return TF.to_tensor(Image.fromarray(blurry))

    def __getitem__(self, index):
        path = self.data_list[index]
        data = np.load(path, allow_pickle=True)
        left_frames = data['left_frames'].item()
        right_frames = data['right_frames'].item()

        left_m = self.left_m if self.left_m != 0 else np.random.randint(1, len(left_frames))
        right_m = self.right_m if self.right_m != 0 else np.random.randint(1, len(right_frames))
        frame0 = self.synthesize_blur(left_frames, left_m)
        frame1 = self.synthesize_blur(right_frames, right_m)

        sharp_frames = [left_frames[i] for i in range(len(left_frames))] + \
                       [right_frames[i] for i in range(len(right_frames))]
        gt = torch.stack([TF.to_tensor(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in sharp_frames])

        left_event = data['left_event'].item()
        right_event = data['right_event'].item()
        xs, ys, ts, ps = [torch.from_numpy(np.concatenate([left_event[k], right_event[k]]).astype(np.float32))
                          for k in ('x', 'y', 't', 'p')]
        events = events_to_stack(xs, ys, ts, ps, self.num_bins, sensor_size=frame0.shape[1:])

        return {
            'frame0': frame0,
            'frame1': frame1,
            'events': events,
            'gt': gt,
            'left_m': left_m,
            'right_m': right_m,
            'path': path,
        }
