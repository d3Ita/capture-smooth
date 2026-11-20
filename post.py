"""Image finishing shared by the CUDA engine and the ONNX (DirectML) export: HUD protection and sharpening.

Both run on the graphics card, in the frames' own precision (half on most cards). Inputs are RGB tensors
in 0-1, shape (1, 3, H, W), already cropped.
"""
import torch
import torch.nn.functional as F

# HUD protection: a minimap, health bar or fixed text stays put for seconds, while scenery moves as soon as
# the camera does. A block counts as HUD only once it has stayed the same for HUD_FRAMES real frames in a
# row; then the in-between copies it from the real frame (the network can smear nearby motion over it).
# A first version compared only the two current frames: during fast camera turns it took a quarter of the
# picture (plain sky, walls) for HUD and pasted real-frame patches into the in-between, which doubled poles
# and wires. Measured on fast GTA 6 turns.
# The test uses the reduced grayscale copies made for the duplicate test (1/8 size, 0-255 levels).
HUD_LEVEL = 3      # a block "stays the same" if its average brightness moves by less than this
HUD_FRAMES = 4     # real frames in a row (~130 ms at 30 fps)

# Sharpening (contrast-adaptive, like AMD CAS / NVIDIA NIS), on brightness only so colours stay as they
# are: l + a * (l - blur3x3), clamped to the 3x3 neighbourhood's min and max so edges get crisper without
# halos. SHARP_MAX is the strength at 100%.
SHARP_MAX = 1.5


def hud_count(count, small_a, small_b):
    """count: blocks' number of real frames in a row without change (1, 1, h/8, w/8), updated for a new pair.
    A block also needs its 3x3 neighbours unchanged, so moving edges never count."""
    same = ((small_b - small_a).abs() < HUD_LEVEL).to(count.dtype)
    same = -F.max_pool2d(-same, 3, 1, 1)
    return (count + 1) * same


def hud_mask_small(count):
    return (count >= HUD_FRAMES).to(count.dtype)


def apply_hud(x, b, mask_small):
    """Copies the protected blocks from the real frame b, with soft edges (mask upscaled bilinearly)."""
    m = F.interpolate(mask_small.to(x.dtype), size=x.shape[-2:], mode="bilinear", align_corners=False)
    return x + (b - x) * m


def sharpen(x, amount):
    """amount: tensor, 0 = off, SHARP_MAX = 100%."""
    luma = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    p = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    blur = F.avg_pool2d(p, 3, 1)
    hi = F.max_pool2d(p, 3, 1)
    lo = -F.max_pool2d(-p, 3, 1)
    sharp = torch.minimum(torch.maximum(luma + (luma - blur) * amount, lo), hi)
    return x + (sharp - luma)
