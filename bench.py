"""Check the model loads, interpolates correctly, and how fast it runs."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from rife import load_model

torch.backends.cudnn.benchmark = True
net = load_model()

# Correctness: a square moving 40 px right should land at +20 px at t=0.5.
H, W = 448, 640
a = np.zeros((H, W, 3), np.float32); a[180:260, 200:280] = 1
b = np.zeros((H, W, 3), np.float32); b[180:260, 240:320] = 1
ta = torch.from_numpy(a).permute(2, 0, 1)[None].cuda().half()
tb = torch.from_numpy(b).permute(2, 0, 1)[None].cuda().half()
t = torch.full((1, 1, H, W), 0.5, device="cuda", dtype=torch.half)
with torch.inference_mode():
    mid = net(ta, tb, t, 1.0)[0, 0].float().cpu().numpy()
cols = np.where(mid[220] > 0.5)[0]
print(f"square at t=0.5: x={cols.min()}..{cols.max()} (expected ~220..299)")

# Speed at the sizes the app uses (padded to a multiple of 64).
for (h, w), scale in [((1472, 2560), 1.0), ((1472, 2560), 0.5), ((1088, 1920), 1.0), ((2176, 3840), 0.5)]:
    x0 = torch.rand(1, 3, h, w, device="cuda", dtype=torch.half)
    x1 = torch.rand(1, 3, h, w, device="cuda", dtype=torch.half)
    tt = torch.full((1, 1, int(h * scale), int(w * scale)), 0.5, device="cuda", dtype=torch.half)  # flow resolution
    with torch.inference_mode():
        for _ in range(5):
            net(x0, x1, tt, scale)
        torch.cuda.synchronize()
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            net(x0, x1, tt, scale)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{w}x{h} scale={scale}: {ms:.1f} ms per generated frame (max {1000/ms:.0f} frames/s)")
