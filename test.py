"""Evaluate AFB on GoPro under symmetric (fixed m) or random (RandEx) exposure."""
import argparse
import os
import random

import lpips
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data import GoProDataset
from models import AFB
from utils.metrics import AverageMeter, evaluate_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='dataset root containing test/<video>/*.npz')
    parser.add_argument('--checkpoint', required=True, help='path to the pretrained AFB weights')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='target frames per input pair = 2 x shutter period (20 for 10x down, 32 for 16x down)')
    parser.add_argument('--left_m', type=int, default=0,
                        help='number of sharp frames averaged into I0 (0: random, RandEx)')
    parser.add_argument('--right_m', type=int, default=0,
                        help='number of sharp frames averaged into I1 (0: random, RandEx)')
    parser.add_argument('--video_name', default=None, help='evaluate a single test video only')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--multi_gpu', action='store_true', help='use nn.DataParallel over all visible GPUs')
    parser.add_argument('--save_dir', default=None, help='directory to write metrics.txt (and images)')
    parser.add_argument('--save_images', action='store_true', help='also save restored frames to --save_dir')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(model, path):
    state_dict = torch.load(path, map_location='cpu')
    state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


def main():
    args = parse_args()
    print(args)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.save_images and args.save_dir is None:
        raise ValueError('--save_images requires --save_dir')

    dataset = GoProDataset(args.data_dir, num_bins=args.num_frames, split='test', left_m=args.left_m,
                           right_m=args.right_m, video_name=args.video_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, generator=torch.Generator().manual_seed(args.seed))

    model = AFB(num_bins=args.num_frames)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()
    if args.multi_gpu:
        model = nn.DataParallel(model)
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    psnrs = [AverageMeter() for _ in range(args.num_frames)]
    ssims = [AverageMeter() for _ in range(args.num_frames)]
    lpipses = [AverageMeter() for _ in range(args.num_frames)]

    with torch.no_grad():
        for batch in tqdm(loader, unit='batch'):
            frame0 = batch['frame0'].to(device)
            frame1 = batch['frame1'].to(device)
            events = batch['events'].to(device)
            gt = batch['gt'].to(device)
            batch_size = frame0.size(0)

            for t in range(args.num_frames):
                target = torch.full((batch_size,), t, dtype=torch.long)
                pred = model(frame0, frame1, events, target)
                evaluate_frame(pred, gt[:, t], psnrs[t], ssims[t], lpipses[t], lpips_fn)

                if args.save_images:
                    for b, path in enumerate(batch['path']):
                        video = os.path.basename(os.path.dirname(path))
                        name = os.path.splitext(os.path.basename(path))[0]
                        os.makedirs(os.path.join(args.save_dir, 'images', video), exist_ok=True)
                        save_image(pred[b], os.path.join(args.save_dir, 'images', video, f'{name}_{t:02d}.png'))

    avg = lambda meters: sum(m.avg for m in meters) / len(meters)
    lines = [f'frame {t:02d}  PSNR {psnrs[t].avg:.4f}  SSIM {ssims[t].avg:.4f}  LPIPS {lpipses[t].avg:.4f}'
             for t in range(args.num_frames)]
    lines.append(f'average   PSNR {avg(psnrs):.4f}  SSIM {avg(ssims):.4f}  LPIPS {avg(lpipses):.4f}')
    print('\n'.join(lines))

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
            f.write(str(args) + '\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
