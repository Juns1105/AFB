import torch
import torch.nn as nn

from .modules import DecoderBlock, EventEncoderBlock, FrameEncoderBlock, PositionalEncoding, conv3x3
from .refine import RefineNet
from .tes import TES
from .tim import TIM


class AFB(nn.Module):
    """Adaptive Feature Blending network for event-guided exposure-agnostic VFI (Fig. 2).

    Args:
        num_bins: number of temporal bins N of the stacked events E_N. It equals the number
            of target timestamps, i.e. twice the shutter period (20 for 10x down, 32 for 16x down).
        in_chn: number of image channels.
        event_chn: number of channels of the events sampled by TES.
        wf: base feature width.
        depth: number of encoder scales.
        num_heads: attention heads per scale.
        relu_slope: negative slope of LeakyReLU.
        t_dim: dimension of the positional encoding of tau.
    """

    def __init__(self, num_bins=32, in_chn=3, event_chn=7, wf=64, depth=3, num_heads=(1, 2, 4),
                 relu_slope=0.2, t_dim=10):
        super().__init__()
        self.t_embedding = PositionalEncoding(num_bins, t_dim)
        self.tes = TES(num_bins, event_chn, wf, t_dim, relu_slope)

        self.frame_stem = nn.Conv2d(in_chn, wf, kernel_size=3, stride=1, padding=1)
        self.event_stem = nn.Conv2d(event_chn, wf, kernel_size=3, stride=1, padding=1)

        self.frame_encoder = nn.ModuleList()
        self.event_encoder = nn.ModuleList()
        self.tim = nn.ModuleList()
        prev_chn = wf
        for i in range(depth):
            chn = wf * 2 ** i
            downsample = i < depth - 1
            self.frame_encoder.append(FrameEncoderBlock(prev_chn, chn, downsample, relu_slope, num_heads[i]))
            self.event_encoder.append(EventEncoderBlock(prev_chn, chn, downsample, relu_slope))
            # The first TIM takes the single-channel E_tau at full resolution.
            self.tim.append(TIM(1 if i == 0 else prev_chn, chn, i > 0, num_heads[i], t_dim, relu_slope))
            prev_chn = chn

        self.decoder = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            chn = wf * 2 ** i
            self.decoder.append(DecoderBlock(prev_chn, chn, relu_slope, t_dim))
            self.skip_convs.append(conv3x3(chn, chn))
            prev_chn = chn

        self.out_conv = conv3x3(prev_chn, in_chn)
        self.refine = RefineNet(in_chn, in_chn)

    def encode_events(self, events):
        feats = []
        x = self.event_stem(events)
        for block in self.event_encoder:
            x, feat = block(x)
            feats.append(feat)
        return feats

    def encode_frame(self, frame, event_feats):
        feats = []
        x = self.frame_stem(frame)
        for block, event_feat in zip(self.frame_encoder, event_feats):
            x, feat = block(x, event_feat)
            feats.append(feat)
        return feats

    def forward(self, frame0, frame1, events, t):
        """
        Args:
            frame0, frame1: (B, 3, H, W) consecutive blurry frames I0, I1 in [0, 1].
            events: (B, N, H, W) events stacked over both shutter periods, E_N.
            t: (B,) integer target timestamps tau in [0, N).
        Returns:
            (B, 3, H, W) restored sharp frame at tau.
        """
        t = t.to(events.device).long()
        t_emb = self.t_embedding(t)

        sampled0, sampled1 = self.tes(events, frame0, frame1, t_emb)
        event_feats0 = self.encode_events(sampled0)
        event_feats1 = self.encode_events(sampled1)
        feats0 = self.encode_frame(frame0, event_feats0)
        feats1 = self.encode_frame(frame1, event_feats1)

        # Adaptive feature blending guided by TIM at every scale.
        x = events[torch.arange(events.size(0), device=events.device), t].unsqueeze(1)  # E_tau
        blended = []
        for tim, event_feat0, event_feat1, feat0, feat1 in zip(self.tim, event_feats0, event_feats1, feats0, feats1):
            weight, x = tim(x, t_emb, event_feat0, event_feat1)
            blended.append(weight * feat0 + (1 - weight) * feat1)

        x = blended[-1]
        for i, (block, skip_conv) in enumerate(zip(self.decoder, self.skip_convs)):
            x = block(x, skip_conv(blended[-i - 2]), t_emb)

        return torch.clamp(self.refine(self.out_conv(x)), 0, 1)
