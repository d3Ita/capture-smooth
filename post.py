"""Image finishing shared by the CUDA engine and the ONNX (DirectML) export: HUD protection and sharpening.

Both run on the graphics card, in the frames' own precision (half on most cards). Inputs are RGB tensors
in 0-1, shape (1, 3, H, W), already cropped.
"""
import torch
import torch.nn.functional as F

# HUD protection: where a pixel is the same in both real frames (minimap, health bar, subtitles over a
# still background...), the right in-between value is that pixel itself. The network can still smear
# nearby motion over it, so those zones are copied from the real frame. "Same" = every channel within
# HUD_LEVEL/255 (capture noise stays under ~8/255), in a whole HUD_SIZE x HUD_SIZE neighbourhood.
HUD_LEVEL = 10
HUD_SIZE = 5

# Sharpening (contrast-adaptive, like AMD CAS / NVIDIA NIS), on brightness only so colours stay as they
# are: l + a * (l - blur3x3), clamped to the 3x3 neighbourhood's min and max so edges get crisper without
# halos. SHARP_MAX is the strength at 100%.
SHARP_MAX = 1.5


def hud_mask(a, b):
    """1 where the two real frames match over a whole neighbourhood, with a soft edge; shape (1, 1, H, W).
    Computed once per pair of real frames."""
    same = ((b - a).abs().amax(1, keepdim=True) < HUD_LEVEL / 255).to(a.dtype)
    m = -F.max_pool2d(-same, HUD_SIZE, 1, HUD_SIZE // 2)
    return F.avg_pool2d(m, 3, 1, 1)


def protect_hud(x, a, b, on):
    """x: in-between frame, a/b: the two real frames, on: 0 or 1 (tensor)."""
    return x + (b - x) * (hud_mask(a, b) * on)


def sharpen(x, amount):
    """amount: tensor, 0 = off, SHARP_MAX = 100%."""
    luma = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    p = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    blur = F.avg_pool2d(p, 3, 1)
    hi = F.max_pool2d(p, 3, 1)
    lo = -F.max_pool2d(-p, 3, 1)
    sharp = torch.minimum(torch.maximum(luma + (luma - blur) * amount, lo), hi)
    return x + (sharp - luma)
