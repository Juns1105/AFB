"""Building blocks shared by the AFB network.

The encoder blocks and the cross-modal fusion (event-image channel attention)
follow EFNet: https://github.com/AHupuJR/EFNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def conv1x1(in_chn, out_chn, bias=True):
    return nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=bias)


def conv3x3(in_chn, out_chn, bias=True):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)


def conv_down(in_chn, out_chn, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding of the integer target timestamp tau."""

    def __init__(self, max_timesteps, dim=10, n=10000):
        super().__init__()
        i = torch.arange(dim // 2, dtype=torch.float32)
        k = torch.arange(max_timesteps, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(max_timesteps, dim)
        pe[:, 0::2] = torch.sin(k / (n ** (2 * i / dim)))
        pe[:, 1::2] = torch.cos(k / (n ** (2 * i / dim)))
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, t):
        return self.pe[t]


class LayerNorm2d(nn.Module):
    """Layer normalization over the channel dimension of a (B, C, H, W) tensor."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class ChannelCrossAttention(nn.Module):
    """Multi-head channel-wise attention with queries from `x` and keys/values from `y`."""

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = conv1x1(dim, dim, bias=bias)
        self.k = conv1x1(dim, dim, bias=bias)
        self.v = conv1x1(dim, dim, bias=bias)
        self.project_out = conv1x1(dim, dim, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'query and key/value feature maps must have the same shape'
        _, _, h, w = x.shape

        q = rearrange(self.q(x), 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(self.k(y), 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(self.v(y), 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CrossModalFusion(nn.Module):
    """Event-image channel attention fusion ('F' in Fig. 2), adopted from EFNet."""

    def __init__(self, dim, num_heads, ffn_expansion_factor=4):
        super().__init__()
        self.norm_frame = LayerNorm2d(dim)
        self.norm_event = LayerNorm2d(dim)
        self.attn = ChannelCrossAttention(dim, num_heads)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * ffn_expansion_factor))

    def forward(self, frame, event):
        h, w = frame.shape[-2:]
        x = frame + self.attn(self.norm_frame(frame), self.norm_event(event))
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.ffn(self.norm_ffn(x))
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class ResidualConvBlock(nn.Module):
    """Two 3x3 conv + LeakyReLU layers with a 1x1 residual projection."""

    def __init__(self, in_chn, out_chn, relu_slope):
        super().__init__()
        self.identity = conv1x1(in_chn, out_chn)
        self.conv_1 = conv3x3(in_chn, out_chn)
        self.conv_2 = conv3x3(out_chn, out_chn)
        self.act = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        return self.act(self.conv_2(self.act(self.conv_1(x)))) + self.identity(x)


class FrameEncoderBlock(ResidualConvBlock):
    """Frame encoder stage that fuses event features before downsampling.

    Returns (input for the next stage, fused feature at this scale).
    """

    def __init__(self, in_chn, out_chn, downsample, relu_slope, num_heads):
        super().__init__(in_chn, out_chn, relu_slope)
        self.fusion = CrossModalFusion(out_chn, num_heads)
        self.downsample = conv_down(out_chn, out_chn) if downsample else None

    def forward(self, x, event_feat):
        x = self.fusion(super().forward(x), event_feat)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class EventEncoderBlock(ResidualConvBlock):
    """Event encoder stage.

    Returns (input for the next stage or None at the last stage, event feature at this scale).
    """

    def __init__(self, in_chn, out_chn, downsample, relu_slope):
        super().__init__(in_chn, out_chn, relu_slope)
        self.out_conv = conv1x1(out_chn, out_chn)
        self.downsample = conv_down(out_chn, out_chn) if downsample else None

    def forward(self, x):
        x = super().forward(x)
        down = self.downsample(x) if self.downsample is not None else None
        return down, self.out_conv(x)


class DecoderBlock(ResidualConvBlock):
    """Upsampling stage conditioned on the positional encoding of tau."""

    def __init__(self, in_chn, out_chn, relu_slope, t_dim=10):
        super().__init__(in_chn, out_chn, relu_slope)
        self.up = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2, bias=True)
        self.t_proj = nn.Sequential(nn.LeakyReLU(relu_slope, inplace=False), nn.Linear(t_dim, 2 * out_chn))

    def forward(self, x, skip, t_emb):
        x = torch.cat([self.up(x), skip], dim=1) + self.t_proj(t_emb)[:, :, None, None]
        return super().forward(x)
