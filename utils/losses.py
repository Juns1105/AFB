import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier loss, Eq. (9): sqrt(||I_gt - I_pred||^2 + eps^2)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target).pow(2) + self.eps ** 2).mean()
