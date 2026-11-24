"""Builds the app's logo files from the logo picture (assets/logo_source.png: a green wave flowing through three
frames, generated with Prisme / Qwen-Image):

    assets/logo.png   the whole logo, black made transparent, trimmed (settings window, wide places)
    assets/icon.ico   Windows icon 16-256 px: a square crop on the middle frame, on a black rounded tile
                      (the whole logo is too wide to read at 16 or 32 px)
    assets/icon.png   the same icon, 256 px

    python tools/make_logo.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def black_to_alpha(rgb):
    """Black background -> transparent, keeping the glow: alpha = brightest channel, colour un-premultiplied."""
    a = rgb.max(axis=2).astype(np.float32)
    col = np.where(a[..., None] > 0, rgb.astype(np.float32) * 255.0 / np.maximum(a[..., None], 1), 0)
    return np.dstack([np.clip(col, 0, 255), a]).astype(np.uint8)


def trim(rgba, pad=12, level=24):
    ys, xs = np.nonzero(rgba[..., 3] > level)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, rgba.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, rgba.shape[1])
    return rgba[y0:y1, x0:x1]


def icon(src, size=256):
    """Middle frame and the wave through it, on a black tile with rounded corners."""
    h, w = src.shape[:2]
    rgba = black_to_alpha(src)
    ys, xs = np.nonzero(rgba[..., 3] > 24)
    cy = (ys.min() + ys.max()) // 2           # vertical centre of the drawing
    half = int((ys.max() - ys.min()) * 0.62)   # a square a bit taller than the wave band
    cx = w // 2
    crop = Image.fromarray(src[cy - half:cy + half, cx - half:cx + half]).resize((size, size), Image.LANCZOS)
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 6, fill=255)
    tile.paste(crop.convert("RGBA"), (0, 0), mask)
    return tile


def main():
    src = np.array(Image.open(ASSETS / "logo_source.png").convert("RGB"))
    Image.fromarray(trim(black_to_alpha(src))).save(ASSETS / "logo.png", optimize=True)
    ic = icon(src)
    ic.save(ASSETS / "icon.png", optimize=True)
    ic.save(ASSETS / "icon.ico", sizes=[(256, 256), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)])
    print("assets/logo.png, assets/icon.png, assets/icon.ico")


if __name__ == "__main__":
    main()
