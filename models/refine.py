"""Refinement network appended to the final decoder stage.

Adapted from EVDI: https://github.com/XiangZ-0/EVDI
"""
import torch
import torch.nn as nn


def conv_relu(in_chn, out_chn):
    return nn.Sequential(nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))


class ChannelAttention(nn.Module):
    def __init__(self, dim, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(dim, dim // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(dim // ratio, dim, 1, bias=False))

    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        x = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        return torch.sigmoid(self.conv1(x))


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class RefineNet(nn.Module):
    def __init__(self, in_chn=3, out_chn=3):
        super().__init__()
        self.in_conv = conv_relu(in_chn, 16)
        self.pre = nn.ModuleList([conv_relu(16, 32), conv_relu(32, 64)])
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.post = nn.ModuleList([conv_relu(128, 32), conv_relu(64, 16)])
        self.out_conv = nn.Conv2d(16, out_chn, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        skips = []
        for layer in self.pre:
            x = layer(x)
            skips.append(x)
        x = self.res2(self.res1(x))
        x = self.ca(x) * x
        x = self.sa(x) * x
        for layer, skip in zip(self.post, reversed(skips)):
            x = layer(torch.cat([x, skip], dim=1))
        return self.out_conv(x)
