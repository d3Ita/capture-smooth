"""RIFE 4.6 frame interpolation network (MIT licence, hzwer/Practical-RIFE).

Rebuilt to match the layers of models/flownet_v4.6.pkl.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=True), nn.LeakyReLU(0.2, True))


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.conv0 = nn.Sequential(conv(cin, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow, scale):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5]


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7, 192)
        self.block1 = IFBlock(12, 128)
        self.block2 = IFBlock(12, 96)
        self.block3 = IFBlock(12, 64)
        self._grid = {}

    def warp(self, img, flow):
        key = (flow.shape, flow.dtype)
        if key not in self._grid:
            n, _, h, w = flow.shape
            gx = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype).view(1, 1, 1, w).expand(n, -1, h, -1)
            gy = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype).view(1, 1, h, 1).expand(n, -1, -1, w)
            norm = torch.tensor([2.0 / (w - 1), 2.0 / (h - 1)], device=flow.device, dtype=flow.dtype).view(1, 2, 1, 1)
            self._grid[key] = (torch.cat([gx, gy], 1), norm)
        grid, norm = self._grid[key]
        g = (grid + flow * norm).permute(0, 2, 3, 1)
        return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)

    def flow(self, img0, img1, timestep):
        """Flow and mask estimated at the resolution of img0/img1 (the reference RIFE 4.6 loop, scale 1)."""
        blocks = [self.block0, self.block1, self.block2, self.block3]
        flow = mask = None
        w0, w1 = img0, img1
        for blk, s in zip(blocks, (8, 4, 2, 1)):
            if flow is None:
                flow, mask = blk(torch.cat((img0, img1, timestep), 1), None, s)
            else:
                f, m = blk(torch.cat((w0, w1, timestep, mask), 1), flow, s)
                flow, mask = flow + f, mask + m
            if blk is not self.block3:  # the last warp is done by the caller, at full resolution
                w0 = self.warp(img0, flow[:, :2])
                w1 = self.warp(img1, flow[:, 2:4])
        return flow, mask

    @staticmethod
    def small(img, scale):
        """Reduced copy of a frame for the flow estimation (computed once per frame, reused for the next pair)."""
        if scale == 1.0:
            return img
        return F.interpolate(img, scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, img0, img1, timestep, scale=1.0, small0=None, small1=None):
        """img0/img1: (1,3,H,W) in [0,1]; timestep: (1,1,h,w) filled with t, at the flow resolution (H*scale).

        With scale < 1 the whole flow estimation runs on reduced frames, and only the final warp and blend
        touch full-resolution pixels. The reference code estimates at the same reduced scales but keeps
        upsampling, warping and concatenating full-resolution images between blocks, which costs more than
        the network itself at 1440p.
        """
        s0 = small0 if small0 is not None else self.small(img0, scale)
        s1 = small1 if small1 is not None else self.small(img1, scale)
        flow, mask = self.flow(s0, s1, timestep)
        if scale != 1.0:
            flow = F.interpolate(flow, scale_factor=1 / scale, mode="bilinear", align_corners=False) / scale
            mask = F.interpolate(mask, scale_factor=1 / scale, mode="bilinear", align_corners=False)
        mask = torch.sigmoid(mask)
        w0 = self.warp(img0, flow[:, :2])
        w1 = self.warp(img1, flow[:, 2:4])
        return w0 * mask + w1 * (1 - mask)


def load_model(path=None, device="cuda", half=True):
    path = path or Path(__file__).with_name("models") / "flownet_v4.6.pkl"
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    net = IFNet()
    net.load_state_dict(sd, strict=True)
    net = net.eval().to(device)
    if half:
        net = net.half()
    for p in net.parameters():
        p.requires_grad_(False)
    return net
