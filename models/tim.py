import torch
import torch.nn as nn

from .modules import ChannelCrossAttention, LayerNorm2d, ResidualConvBlock, conv3x3, conv_down


class EventCrossAttention(nn.Module):
    """Channel-wise attention of Eq. (7): sampled-event features attend to E_tau features."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_query = LayerNorm2d(dim)
        self.norm_kv = LayerNorm2d(dim)
        self.attn = ChannelCrossAttention(dim, num_heads)

    def forward(self, query, key_value):
        return self.attn(self.norm_query(query), self.norm_kv(key_value))


class TIM(ResidualConvBlock):
    """Target-adaptive Importance Mapping (Sec. 3.3, Fig. 3(b)) at one encoder scale.

    Encodes the tau-centered event stack E_tau, attends to it with the features of the
    sampled events from TES (Eq. 7), injects the positional encoding of tau, and predicts
    the importance map w_tau that blends the fused features (Eq. 8):
        F_tau = w_tau * F0 + (1 - w_tau) * F1

    Returns (w_tau, E_tau feature passed to the next scale).
    """

    def __init__(self, in_chn, out_chn, downsample, num_heads, t_dim=10, relu_slope=0.2):
        super().__init__(in_chn, out_chn, relu_slope)
        self.downsample = conv_down(out_chn, out_chn) if downsample else None
        self.attention = EventCrossAttention(out_chn, num_heads)
        self.t_proj = nn.Sequential(nn.LeakyReLU(relu_slope, inplace=False), nn.Linear(t_dim, 2 * out_chn))
        self.head_conv_1 = conv3x3(2 * out_chn, out_chn)
        self.head_conv_2 = conv3x3(out_chn, out_chn)

    def forward(self, x, t_emb, event_feat0, event_feat1):
        x = super().forward(x)
        if self.downsample is not None:
            x = self.downsample(x)

        attended0 = self.attention(event_feat0, x)
        attended1 = self.attention(event_feat1, x)
        y = self.t_proj(t_emb)[:, :, None, None] + torch.cat([attended0, attended1], dim=1)

        weight = torch.sigmoid(self.head_conv_2(self.act(self.head_conv_1(y))))
        return weight, x
