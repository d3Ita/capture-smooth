"""Exports the RIFE model to ONNX for the DirectML engine (one file per resolution).

Each file takes two BGR frames (H, W, 3 bytes) and a time t, and returns the in-between frame as RGBA bytes,
so the colour conversion, padding and the network all run on the graphics card.

    python tools/export_onnx.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engines import onnx_name, padded, post_name  # noqa: E402
from post import protect_hud, sharpen  # noqa: E402
from rife import load_model  # noqa: E402

SIZES = [(1920, 1080, 1.0), (1920, 1080, 0.5), (2560, 1440, 0.5)]


_GRIDS = {}


def warp_onnx(img, flow):
    """Same as IFNet.warp, with an explicit float32 grid: GridSample only runs on the graphics card (and
    exists at all in ONNX Runtime) for float32 inputs."""
    # While tracing, shape values are traced tensors: int() turns them back into plain numbers for the cache.
    n, h, w = int(flow.shape[0]), int(flow.shape[2]), int(flow.shape[3])
    if (h, w) not in _GRIDS:  # built outside the trace: small float32 constants (one row, one column)
        gx = torch.linspace(-1.0, 1.0, w, device=flow.device).view(1, 1, 1, w)
        gy = torch.linspace(-1.0, 1.0, h, device=flow.device).view(1, 1, h, 1)
        _GRIDS[(h, w)] = (gx, gy)
    gx, gy = _GRIDS[(h, w)]
    # Broadcasting the row and the column against the flow keeps the full-size grid out of the file.
    f = flow.float()
    g = torch.cat([f[:, 0:1] * (2.0 / (w - 1)) + gx, f[:, 1:2] * (2.0 / (h - 1)) + gy], 1).permute(0, 2, 3, 1)
    out = F.grid_sample(img.float(), g, mode="bilinear", padding_mode="border", align_corners=True)
    return out.to(img.dtype)


class Interp(nn.Module):
    def __init__(self, net, w, h, scale):
        super().__init__()
        self.net, self.w, self.h, self.scale = net, w, h, scale
        self.ph, self.pw = padded(h, w, scale)
        self.register_buffer("bgr", torch.tensor([2, 1, 0]))

    def prep(self, f):
        x = f.permute(2, 0, 1).unsqueeze(0).index_select(1, self.bgr).half() / 255
        return F.pad(x, (0, self.pw - self.w, 0, self.ph - self.h), mode="replicate")

    def forward(self, frame0, frame1, t, sharp, hud):
        x0, x1 = self.prep(frame0), self.prep(frame1)
        sh, sw = int(self.ph * self.scale), int(self.pw * self.scale)
        ts = t.view(1, 1, 1, 1).expand(1, 1, sh, sw)
        mid = self.net(x0, x1, ts, self.scale)[:, :, :self.h, :self.w]
        a, b = x0[:, :, :self.h, :self.w], x1[:, :, :self.h, :self.w]
        y = sharpen(protect_hud(mid, a, b, hud.view(1, 1, 1, 1).half()), sharp.view(1, 1, 1, 1).half())
        return to_rgba(y.float())


def to_rgba(y):
    rgb = (y[0] * 255 + 0.5).clamp(0, 255)
    alpha = torch.full_like(rgb[:1], 255)
    return torch.cat((rgb, alpha), 0).permute(1, 2, 0).to(torch.uint8)


class Real(nn.Module):
    """A captured frame as shown: BGR bytes -> sharpened RGBA bytes."""

    def __init__(self):
        super().__init__()
        self.register_buffer("bgr", torch.tensor([2, 1, 0]))

    def forward(self, frame, sharp):
        x = frame.permute(2, 0, 1).unsqueeze(0).index_select(1, self.bgr).half() / 255
        return to_rgba(sharpen(x, sharp.view(1, 1, 1, 1).half()).float())


def main():
    net = load_model()
    net.warp = warp_onnx
    out_dir = ROOT / "models"
    for w, h, scale in SIZES:
        m = Interp(net, w, h, scale).cuda().eval()
        f0 = torch.randint(0, 255, (h, w, 3), dtype=torch.uint8, device="cuda")
        f1 = torch.randint(0, 255, (h, w, 3), dtype=torch.uint8, device="cuda")
        t = torch.tensor([0.5], dtype=torch.half, device="cuda")
        sharp = torch.tensor([0.45], dtype=torch.float32, device="cuda")
        hud = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        path = out_dir / onnx_name(w, h, scale)
        with torch.no_grad():
            m(f0, f1, t, sharp, hud)  # fills the warp grid cache for these sizes
        torch.onnx.export(m, (f0, f1, t, sharp, hud), str(path), input_names=["frame0", "frame1", "t", "sharp", "hud"],
                          output_names=["rgba"], opset_version=17, dynamo=False,
                          do_constant_folding=False)  # folding would store the full-size warp grids (100+ Mo)
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} Mo")
    for w, h in sorted({(w, h) for w, h, _ in SIZES}):
        path = out_dir / post_name(w, h)
        f = torch.randint(0, 255, (h, w, 3), dtype=torch.uint8, device="cuda")
        sharp = torch.tensor([0.45], dtype=torch.float32, device="cuda")
        torch.onnx.export(Real().cuda().eval(), (f, sharp), str(path), input_names=["frame", "sharp"],
                          output_names=["rgba"], opset_version=17, dynamo=False)
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} Mo")


if __name__ == "__main__":
    main()
