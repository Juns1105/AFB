import torch
import torch.nn as nn

from .modules import LayerNorm2d, conv1x1, conv3x3


class TES(nn.Module):
    """Target-adaptive Event Sampling (Sec. 3.2, Fig. 3(a)).

    Samples the stacked events E_N around the target timestamp tau and the unknown
    exposure of each blurry frame, i.e. E_{tau->[ts0,te0]} = TES(E_N, I0, tau) and
    E_{tau->[ts1,te1]} = TES(E_N, I1, tau).
    """

    def __init__(self, num_bins, out_chn=7, dim=64, t_dim=10, relu_slope=0.2):
        super().__init__()
        self.conv_event = conv1x1(num_bins, dim)
        self.conv_frame0 = conv1x1(3, dim)
        self.conv_frame1 = conv1x1(3, dim)

        self.norm_event = LayerNorm2d(dim)
        self.norm_frame0 = LayerNorm2d(dim)
        self.norm_frame1 = LayerNorm2d(dim)

        self.t_proj = nn.Sequential(nn.LeakyReLU(relu_slope, inplace=False), nn.Linear(t_dim, dim))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.out_conv0 = conv3x3(dim, out_chn)
        self.out_conv1 = conv3x3(dim, out_chn)

    def forward(self, events, frame0, frame1, t_emb):
        # Event features with the positional encoding of tau added channel-wise.
        f_event = self.t_proj(t_emb)[:, :, None, None] + self.conv_event(events)
        f_event_norm = self.norm_event(f_event)

        # Correlation scores, Eq. (5): S_I = GAP(sigmoid(norm(f_E) * norm(f_I))).
        score0 = self.gap(torch.sigmoid(f_event_norm * self.norm_frame0(self.conv_frame0(frame0))))
        score1 = self.gap(torch.sigmoid(f_event_norm * self.norm_frame1(self.conv_frame1(frame1))))

        return self.out_conv0(f_event * score0), self.out_conv1(f_event * score1)
