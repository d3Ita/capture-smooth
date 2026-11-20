"""Draws assets/icon.ico: three frames in motion (the middle one generated), on a dark rounded tile.

    python tools/make_icon.py
"""
import struct
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def draw(n=256):
    s = 4  # supersampling
    N = n * s
    img = np.zeros((N, N, 4), np.uint8)
    r = int(N * 0.2)
    tile = np.zeros((N, N), np.uint8)
    cv2.rectangle(tile, (r, 0), (N - r, N), 255, -1)
    cv2.rectangle(tile, (0, r), (N, N - r), 255, -1)
    for cx, cy in ((r, r), (N - r, r), (r, N - r), (N - r, N - r)):
        cv2.circle(tile, (cx, cy), r, 255, -1)
    grad = np.linspace(0, 1, N)[:, None]
    bg = (np.array([40, 22, 18]) * (1 - grad)[..., None] + np.array([90, 40, 30]) * grad[..., None])
    img[..., :3] = bg.astype(np.uint8)
    img[..., 3] = tile
    # Three frames: two real (white outline), one generated between them (filled cyan-violet).
    w, h = int(N * 0.22), int(N * 0.40)
    y0 = (N - h) // 2
    for i, (x, kind) in enumerate(((int(N * 0.12), "real"), (int(N * 0.39), "ai"), (int(N * 0.66), "real"))):
        rect = (x, y0, x + w, y0 + h)
        if kind == "ai":
            over = img.copy()
            cv2.rectangle(over, rect[:2], rect[2:], (255, 120, 170, 255), -1)
            img[:] = cv2.addWeighted(over, 0.9, img, 0.1, 0)
            img[..., 3] = tile
        else:
            cv2.rectangle(img, rect[:2], rect[2:], (245, 245, 245, 255), int(N * 0.035))
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)


def write_ico(path, sizes=(256, 64, 48, 32, 16)):
    big = draw(256)
    pngs = [cv2.imencode(".png", cv2.resize(big, (s, s), interpolation=cv2.INTER_AREA))[1].tobytes()
            for s in sizes]
    head = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries = b""
    for s, data in zip(sizes, pngs):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(head + entries + b"".join(pngs))
    cv2.imwrite(str(path.with_suffix(".png")), big)


if __name__ == "__main__":
    write_ico(ROOT / "assets" / "icon.ico")
    print("assets/icon.ico")
