"""Runs a video through Capture Smooth's copy / new frame / scene cut decisions (cadence.py), as the app does,
and reports how many real frames per second it finds. For checking a PS Remote Play (or capture card)
recording: record at the stream's own rate (60 fps), then

    python tools/cadence_check.py recording.mp4 --mode tolerant [--jitter 8] [--labels labels.npy]

--labels: optional ground truth, one byte per frame, 1 = new frame (the first frame counts as new).
--jitter: adds random arrival delays (ms, half-normal) like a network would.
"""
import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cadence import Cadence, SceneCuts  # noqa: E402
from engines import change_cpu, small_gray  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--mode", choices=["strict", "tolerant"], default="tolerant")
    ap.add_argument("--labels", help="vérité terrain (.npy, 1 = nouvelle image)")
    ap.add_argument("--jitter", type=float, default=0, help="gigue d'arrivée simulée, en ms")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    c = av.open(args.video)
    stream = c.streams.video[0]
    fps = float(stream.average_rate or 60)
    rng = np.random.default_rng(args.seed)
    cadence, cuts = Cadence(args.mode == "tolerant", 1 / fps), SceneCuts()
    new, cut_at, prev_small, t_last = [], [], None, 0.0
    for i, frame in enumerate(c.decode(video=0)):
        t = max(i / fps + abs(rng.normal(0, args.jitter / 1000)) if args.jitter else i / fps, t_last)
        t_last = t
        small = small_gray(frame.to_ndarray(format="bgr24"))
        if prev_small is None:
            cadence.accept(1.0, t)
            is_new = True
        else:
            is_new = not cadence.is_dup(change_cpu(prev_small, small), t)
            if is_new:
                a = cv2.resize(prev_small, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
                b = cv2.resize(small, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
                if cuts.is_cut(float(np.abs(b - a).mean())):
                    cut_at.append(i)
        if is_new:
            prev_small = small  # copies are compared with the last new frame, as in the app
        new.append(is_new)
    c.close()

    new = np.array(new)
    secs = len(new) / fps
    print(f"{args.video} : {len(new)} images ({fps:.0f} par seconde, {secs:.1f} s), mode {args.mode}"
          + (f", gigue {args.jitter:.0f} ms" if args.jitter else ""))
    print(f"  vraies images trouvées : {new.sum() / secs:.1f} par seconde, période estimée {cadence.period * 1000:.1f} ms")
    print(f"  changements de plan : {len(cut_at)}" + (f" (images {cut_at[:10]})" if cut_at else ""))
    if args.labels:
        lab = np.load(args.labels)[:len(new)].astype(bool)
        new = new[:len(lab)]
        print(f"  vérité : {lab.sum() / secs:.1f} vraies images par seconde ; copies prises pour des nouvelles : "
              f"{(new & ~lab).sum()}, nouvelles prises pour des copies : {(~new & lab).sum()}")


if __name__ == "__main__":
    main()
