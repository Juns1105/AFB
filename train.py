"""Train AFB on GoPro with randomly sampled exposures and target timestamps."""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GoProDataset
from models import AFB
from utils.losses import CharbonnierLoss
from utils.metrics import AverageMeter, calc_psnr, quantize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='dataset root containing train/ and test/')
    parser.add_argument('--save_dir', default='./experiments', help='checkpoints go to <save_dir>/<exp_name>/models')
    parser.add_argument('--exp_name', default='afb_gopro')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='target frames per input pair = 2 x shutter period (20 for 10x down, 32 for 16x down)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--cosine_period', type=int, default=100,
                        help='T_max (in epochs) of the cosine annealing learning rate schedule')
    parser.add_argument('--eps', type=float, default=1e-6, help='epsilon of the Charbonnier loss')
    parser.add_argument('--crop_size', type=int, default=128, help='training crop size')
    parser.add_argument('--left_m', type=int, default=0,
                        help='number of sharp frames averaged into I0 (0: random within the shutter period)')
    parser.add_argument('--right_m', type=int, default=0,
                        help='number of sharp frames averaged into I1 (0: random within the shutter period)')
    parser.add_argument('--val_target', type=int, default=9, help='target timestamp used for validation')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--multi_gpu', action='store_true', help='use nn.DataParallel over all visible GPUs')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    with tqdm(loader, unit='batch', desc=f'Train epoch {epoch}') as pbar:
        for batch in pbar:
            frame0 = batch['frame0'].to(device)
            frame1 = batch['frame1'].to(device)
            events = batch['events'].to(device)
            gt = batch['gt'].to(device)

            pred = model(frame0, frame1, events, batch['t'])
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, target, epoch):
    model.eval()
    total_loss = 0
    psnr = AverageMeter()
    with tqdm(loader, unit='batch', desc=f'Val epoch {epoch}') as pbar:
        for batch in pbar:
            frame0 = batch['frame0'].to(device)
            frame1 = batch['frame1'].to(device)
            events = batch['events'].to(device)
            gt = batch['gt'][:, target].to(device)

            t = torch.full((frame0.size(0),), target, dtype=torch.long)
            pred = model(frame0, frame1, events, t)
            loss = criterion(pred, gt)

            total_loss += loss.item()
            for b in range(gt.size(0)):
                psnr.update(calc_psnr(quantize(pred[b]), quantize(gt[b])))
            pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader), psnr.avg


def main():
    args = parse_args()
    print(args)
    assert 0 <= args.val_target < args.num_frames, '--val_target must be in [0, num_frames)'
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = GoProDataset(args.data_dir, args.num_frames, split='train', left_m=args.left_m,
                             right_m=args.right_m, crop_size=(args.crop_size, args.crop_size))
    val_set = GoProDataset(args.data_dir, args.num_frames, split='test', left_m=args.left_m, right_m=args.right_m)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = AFB(num_bins=args.num_frames).to(device)
    net = nn.DataParallel(model) if args.multi_gpu else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_period)
    criterion = CharbonnierLoss(eps=args.eps)

    ckpt_dir = os.path.join(args.save_dir, args.exp_name, 'models')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_psnr = -float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        val_loss, val_psnr = validate(net, val_loader, criterion, device, args.val_target, epoch)

        print(f'epoch {epoch}: lr {scheduler.get_last_lr()[0]:.2e}  train loss {train_loss:.6f}  '
              f'val loss {val_loss:.6f}  val PSNR {val_psnr:.4f}')

        # Keep the checkpoint whenever the validation PSNR improves.
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            path = os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pth')
            torch.save(model.state_dict(), path)
            print(f'saved {path} (best val PSNR {best_psnr:.4f})')


if __name__ == '__main__':
    main()
