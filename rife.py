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


class Head(nn.Module):
    """Feature encoder of RIFE 4.22+: 4 feature channels per frame, at full resolution."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class IFBlock2(nn.Module):
    """RIFE 4.22+ block: also returns 8 feature channels passed on to the next block."""

    def __init__(self, cin, c):
        super().__init__()
        self.conv0 = nn.Sequential(conv(cin, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow, scale):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet2(IFNet):
    """RIFE 4.22 to 4.26 (Practical-RIFE, MIT): a feature encoder, and 4 or 5 blocks. Same interface and the
    same speed-up as IFNet: flow estimated on reduced frames, only the final warp at full resolution."""

    def __init__(self, widths):
        nn.Module.__init__(self)
        self.encode = Head()
        self.blocks = nn.ModuleList([IFBlock2(15 if i == 0 else 28, c) for i, c in enumerate(widths)])
        self.scales = [2 ** (len(widths) - 1 - i) for i in range(len(widths))]  # 16,8,4,2,1 or 8,4,2,1
        self.align = 4 * self.scales[0]  # flow resolution must divide by this
        self._grid = {}

    def flow(self, img0, img1, timestep):
        f0, f1 = self.encode(img0), self.encode(img1)
        flow = mask = feat = None
        w0, w1 = img0, img1
        last = len(self.blocks) - 1
        for i, (blk, s) in enumerate(zip(self.blocks, self.scales)):
            if flow is None:
                flow, mask, feat = blk(torch.cat((img0, img1, f0, f1, timestep), 1), None, s)
            else:
                wf0, wf1 = self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:4])
                fd, mask, feat = blk(torch.cat((w0, w1, wf0, wf1, timestep, mask, feat), 1), flow, s)
                flow = flow + fd
            if i != last:
                w0 = self.warp(img0, flow[:, :2])
                w1 = self.warp(img1, flow[:, 2:4])
        return flow, mask


def load_model(path=None, device="cuda", half=True):
    path = path or Path(__file__).with_name("models") / "flownet_v4.6.pkl"
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if any(k.startswith("encode.") for k in sd):  # RIFE 4.22+; the training-only parts are not needed
        sd = {k: v for k, v in sd.items() if not k.startswith(("teacher.", "caltime."))}
        n = len({k.split(".")[0] for k in sd if k.startswith("block")})
        widths = [sd[f"block{i}.conv0.1.0.weight"].shape[0] for i in range(n)]
        sd = {(f"blocks.{k[5]}{k[6:]}" if k.startswith("block") else k): v for k, v in sd.items()}
        net = IFNet2(widths)
    else:
        net = IFNet()
        net.align = 32
    net.load_state_dict(sd, strict=True)
    net = net.eval().to(device)
    if half:
        net = net.half()
    for p in net.parameters():
        p.requires_grad_(False)
    return net
