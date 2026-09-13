import math

from pytorch_msssim import ssim


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def quantize(img):
    """[0, 1] image -> quantized [0, 255] image."""
    return img.mul(255).clamp(0, 255).round()


def calc_psnr(pred, gt):
    """PSNR between quantized [0, 255] images."""
    mse = (pred - gt).div(255).pow(2).mean() + 1e-8
    return -10 * math.log10(mse)


def evaluate_frame(pred, gt, psnr_meter, ssim_meter, lpips_meter, lpips_fn):
    """Accumulate PSNR / SSIM / LPIPS of a batch of predictions for one target timestamp."""
    for b in range(gt.size(0)):
        q_pred = quantize(pred[b])
        q_gt = quantize(gt[b])
        psnr_meter.update(calc_psnr(q_pred, q_gt))
        ssim_meter.update(ssim(q_pred.unsqueeze(0), q_gt.unsqueeze(0), data_range=255, size_average=False).item())
        lpips_meter.update(lpips_fn(pred[b], gt[b]).item())
