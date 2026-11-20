"""Records the capture card for the demo videos: keeps only the real game frames (duplicates dropped),
saved as 1080p JPEGs with their capture time.

    python tools/record_demo.py --seconds 20 --out demo_raw
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engines import dup_cpu, small_gray  # noqa: E402
from smooth import pick_device  # noqa: E402


def is_dup(a, b):
    return dup_cpu(small_gray(a), small_gray(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--out", default=str(ROOT / "demo_raw"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    idx, name = pick_device(args.device)
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_FPS, 60)
    ok, prev = cap.read()
    if not ok:
        sys.exit("Impossible de lire le boîtier (ferme Capture Smooth et OBS).")
    print(f"Enregistrement de {name} pendant {args.seconds:.0f} s… joue et bouge !")

    pool = ThreadPoolExecutor(6)
    frames, total, dups = [], 0, 0
    t0 = time.perf_counter()
    prev = None
    while time.perf_counter() - t0 < args.seconds:
        ok, f = cap.read()
        t = time.perf_counter() - t0
        if not ok:
            continue
        total += 1
        if prev is not None and is_dup(prev, f):
            dups += 1
            continue
        prev = f
        path = out / f"{len(frames):05d}.jpg"
        small = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA)
        pool.submit(cv2.imwrite, str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append({"file": path.name, "t": round(t, 5)})
    pool.shutdown()
    cap.release()
    (out / "frames.json").write_text(json.dumps({"frames": frames, "captured": total, "dups": dups}))
    print(f"{len(frames)} images réelles ({len(frames) / args.seconds:.1f}/s), {dups} doublons, sur {total} reçues.")


if __name__ == "__main__":
    main()
