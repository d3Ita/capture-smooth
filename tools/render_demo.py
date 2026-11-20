"""Builds the before/after media for the GitHub page from a recording made with record_demo.py.

    python tools/render_demo.py --raw demo_raw --out docs/media

Writes:
  compare_60fps.mp4   side by side, real time: original 30 fps | Capture Smooth x2 (60 fps)
  compare_slowmo.mp4  side by side, 1/4 speed: original | x8, so the difference shows on any screen
  frames.jpg          real frame, two AI frames, next real frame (for the README)
  *_poster.jpg        first image of each video
Needs ffmpeg (pip install imageio-ffmpeg) for H.264 video that browsers can play.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rife import load_model  # noqa: E402

GAME_FPS = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class Video:
    def __init__(self, path, w, h, fps=60, crf=21):
        self.p = subprocess.Popen(
            [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)
        self.first = None

    def write(self, img):
        if self.first is None:
            self.first = img.copy()
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self, poster=None):
        self.p.stdin.close()
        self.p.wait()
        if poster is not None and self.first is not None:
            cv2.imwrite(str(poster), self.first, [cv2.IMWRITE_JPEG_QUALITY, 88])


class Interp:
    def __init__(self, frames):
        self.frames = frames
        self.net = load_model()
        self.cache = {}

    def gpu(self, i):
        if i not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            x = torch.from_numpy(self.frames[i]).cuda().permute(2, 0, 1)[None].flip(1).half() / 255
            h, w = x.shape[2:]
            self.cache[i] = F.pad(x, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32), mode="replicate")
        return self.cache[i]

    @torch.inference_mode()
    def __call__(self, i, t):
        if t == 0 or i + 1 >= len(self.frames):
            return self.frames[min(i, len(self.frames) - 1)]
        a, b = self.gpu(i), self.gpu(i + 1)
        ts = torch.full((1, 1, a.shape[2], a.shape[3]), t, device="cuda", dtype=torch.half)
        h, w = self.frames[i].shape[:2]
        out = self.net(a, b, ts, 1.0)[0, :, :h, :w].flip(0).mul(255).add(0.5).clamp(0, 255).byte()
        return out.permute(1, 2, 0).cpu().numpy()


def label(img, text, sub=""):
    cv2.putText(img, text, (24, 46), FONT, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (24, 46), FONT, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (24, 84), FONT, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, sub, (24, 84), FONT, 0.75, (200, 230, 255), 2, cv2.LINE_AA)
    return img


def side(left, right, ltxt, rtxt, lsub="", rsub=""):
    w, h = 958, 540  # two panels + a 4 px divider = 1920
    a = label(cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA), ltxt, lsub)
    b = label(cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA), rtxt, rsub)
    bar = np.full((h, 4, 3), 255, np.uint8)
    return np.hstack([a, bar, b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "demo_raw"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--start", type=float, default=0, help="start, in seconds of game time")
    ap.add_argument("--seconds", type=float, default=12, help="length of the real-time comparison")
    ap.add_argument("--slow-seconds", type=float, default=4, help="game time shown in slow motion")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((raw / "frames.json").read_text())
    first = int(args.start * GAME_FPS)
    n_needed = int(max(args.seconds, args.slow_seconds) * GAME_FPS) + 2
    files = [f["file"] for f in meta["frames"]][first:first + n_needed]
    frames = [cv2.imread(str(raw / f)) for f in files]
    print(f"{len(frames)} images réelles chargées")
    ip = Interp(frames)

    # Real time at 60 fps (x2) and 90 fps (x3): the original holds each game frame for m video frames.
    for m in (2, 3):
        fps = GAME_FPS * m
        n_out = int(args.seconds * fps)
        v = Video(out / f"compare_{fps}fps.mp4", 1920, 540, fps=fps)
        for n in range(min(n_out, (len(frames) - 1) * m)):
            i, k = divmod(n, m)
            v.write(side(frames[i], ip(i, k / m), "Original", "Capture Smooth",
                         "30 fps", f"{fps} fps (x{m}, AI frames in between)"))
        v.close(out / f"compare_{fps}fps_poster.jpg")
        print(f"compare_{fps}fps.mp4")

    # Slow motion, 1/4 speed: 8 video frames per game frame.
    n_out = int(args.slow_seconds * GAME_FPS) * 8
    v = Video(out / "compare_slowmo.mp4", 1920, 540)
    for n in range(min(n_out, (len(frames) - 1) * 8)):
        i, k = divmod(n, 8)
        v.write(side(frames[i], ip(i, k / 8), "Original (1/4 speed)", "Capture Smooth (1/4 speed)",
                     "each game frame held", "7 AI frames between each pair"))
    v.close(out / "compare_slowmo_poster.jpg")
    print("compare_slowmo.mp4")

    # Frame strip: the pair with the most motion, its two in-betweens at x3.
    diffs = [cv2.absdiff(frames[i], frames[i + 1]).mean() for i in range(len(frames) - 1)]
    i = int(np.argsort(diffs)[int(len(diffs) * 0.85)])  # strong but not a camera cut
    panels = [(frames[i], "Real frame"), (ip(i, 1 / 3), "AI frame 1/3"), (ip(i, 2 / 3), "AI frame 2/3"),
              (frames[i + 1], "Next real frame")]
    tiles = []
    for img, txt in panels:
        t = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        cap = np.full((48, 640, 3), 24, np.uint8)
        color = (255, 170, 120) if "AI" in txt else (240, 240, 240)
        cv2.putText(cap, txt, (14, 33), FONT, 0.9, color, 2, cv2.LINE_AA)
        tiles.append(np.vstack([t, cap]))
    gap = np.full((408, 6, 3), 24, np.uint8)
    strip = np.hstack(sum(([t, gap] for t in tiles), [])[:-1])
    cv2.imwrite(str(out / "frames.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("frames.jpg")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()
