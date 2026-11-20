"""AI engines. Both take BGR frames from the capture and return RGBA bytes ready for the screen.

- CudaEngine: PyTorch + CUDA graphs. Fastest, NVIDIA only.
- DmlEngine: ONNX Runtime + DirectML. Any DirectX 12 graphics card (NVIDIA, AMD, Intel).

Common interface:
    upload(frame)        -> handle for one captured frame
    is_dup(prev, cur)    -> True if cur is a copy of prev (30 fps game in a 60 Hz signal, still image)
    tiny(handle)         -> 32x18 grayscale thumbnail (float32 numpy), for scene-cut detection
    set_options(sharpness, hud)  -> sharpening amount (0-1) and HUD protection on/off
    set_pair(prev, cur)  -> the two frames the next in-betweens are made from
    interp(t)            -> in-between frame at time t in (0, 1), RGBA numpy array
    real(cur)            -> the captured frame itself (sharpened like the others), RGBA numpy array
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Duplicate test, on a grayscale copy reduced 8x (each pixel = average of 8x8): the capture noise
# averages out there, while real motion survives even in dark scenes. Measured on a dark GTA 6 scene
# (mean brightness 24/255) through an HD60 X: copies never had a pixel change by 3 levels or more,
# new frames always had at least 0.017% of them. A fixed full-resolution threshold (the previous
# test) missed half the new frames in the dark, which halved the displayed frame rate.
DUP_REDUCE = 8
DUP_LEVEL = 3
DUP_SHARE = 4e-5

# HUD protection and sharpening: see post.py (shared with the ONNX export, so both engines look the same).
SHARP_MAX = 1.5  # same values as in post.py (post.py imports torch, which the DirectML build lacks)
HUD_LEVEL = 3
HUD_FRAMES = 4


def small_gray(frame_bgr):
    """Reduced grayscale copy used by the duplicate test (CPU path)."""
    h, w = frame_bgr.shape[:2]
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (w // DUP_REDUCE, h // DUP_REDUCE), interpolation=cv2.INTER_AREA)


def dup_cpu(small_a, small_b):
    d = cv2.absdiff(small_a, small_b)
    _, m = cv2.threshold(d, DUP_LEVEL - 1, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(m) / d.size < DUP_SHARE


def app_dir():
    """Folder holding models/ (next to the .exe once packaged)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def padded(h, w, scale, align=32):
    """Size the network needs: the flow resolution (h*scale) must divide by the model's align (32 for RIFE 4.6)."""
    pad = int(align / scale)
    return (h + pad - 1) // pad * pad, (w + pad - 1) // pad * pad


class FrameCache:
    """Per-frame values computed once (reduced copies), kept for the last few frames only."""

    def __init__(self, keep=3):
        self.keep, self.items = keep, {}

    def get(self, handle, key, fn):
        entry = self.items.setdefault(id(handle), {"_ref": handle})  # the reference keeps the id unique
        if key not in entry:
            entry[key] = fn(handle)
        while len(self.items) > self.keep:
            self.items.pop(next(iter(self.items)))
        return entry[key]


class GpuFrame:
    """A displayed frame that stays on the graphics card (CUDA -> OpenGL interop), ready once event is done."""
    __slots__ = ("tensor", "event")

    def __init__(self, tensor, event):
        self.tensor, self.event = tensor, event


class CudaEngine:
    name = "CUDA"

    def __init__(self, src_size, work_size, scale, use_graphs=True):
        import torch
        import torch.nn.functional as F
        from rife import load_model

        if not torch.cuda.is_available():
            raise RuntimeError("pas de carte NVIDIA compatible CUDA")
        if os.environ.get("CAPTURE_SMOOTH_NO_CUDA"):  # for testing the DirectML fallback on an NVIDIA PC
            raise RuntimeError("CUDA désactivé (CAPTURE_SMOOTH_NO_CUDA)")
        major, minor = torch.cuda.get_device_capability(0)
        gpu = torch.cuda.get_device_name(0)
        built = {a for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
        # A card runs code built for its own architecture or an older one of the same generation.
        if not any(int(a[3:]) // 10 == major and int(a[3:]) % 10 <= minor for a in built):
            raise RuntimeError(f"{gpu} (sm_{major}{minor}) non prise en charge par cette version de PyTorch")
        # Half precision is fast from Volta / Turing (RTX 20, GTX 16) on; GTX 10 cards run it 64x slower.
        self.dtype = torch.half if major >= 7 else torch.float32
        self.gpu = gpu
        torch.backends.cudnn.benchmark = True
        self.torch, self.F = torch, F
        self.W, self.H = work_size
        self.scale = scale
        self.resize = src_size != work_size
        self.net = load_model(app_dir() / "models" / "flownet_v4.6.pkl", half=self.dtype == torch.half)
        self.ph, self.pw = padded(self.H, self.W, scale, self.net.align)
        self.use_graphs = use_graphs
        self.graphs = {}  # t -> (CUDA graph, its RGBA output)
        # Fixed input slots for the CUDA graphs (a graph always reads the same memory).
        self.slot0 = torch.zeros((1, 3, self.ph, self.pw), device="cuda", dtype=self.dtype)
        self.slot1 = torch.zeros_like(self.slot0)
        # Options live in GPU scalars, so the recorded graphs follow changes made while playing.
        self.sharp = torch.zeros((), device="cuda", dtype=self.dtype)
        self.hud = True
        sh8, sw8 = self.H // DUP_REDUCE, self.W // DUP_REDUCE
        self.hudcount = torch.zeros((1, 1, sh8, sw8), device="cuda", dtype=torch.float32)  # see post.py
        self.hudmask = torch.zeros((1, 1, sh8, sw8), device="cuda", dtype=self.dtype)       # read by the graphs
        self.opts = (None, None)
        self.tsteps = {}
        self.cache = FrameCache()
        # Pinned output buffers, reused round-robin (more than the display queue can hold).
        self.ring = [torch.full((self.H, self.W, 4), 255, dtype=torch.uint8).pin_memory() for _ in range(10)]
        self.ri = 0
        self.rgba = torch.full((self.H, self.W, 4), 255, device="cuda", dtype=torch.uint8)
        self.gpu_frames = False  # set by the display when it can take frames straight from the GPU
        self.gring = None
        self.gi = 0
        self.ctx = torch.inference_mode()

    def use_gpu_frames(self):
        """Frames are handed to the display on the GPU (no download): a ring of RGBA buffers, more than the
        display queue can hold."""
        torch = self.torch
        self.gring = [torch.full((self.H, self.W, 4), 255, device="cuda", dtype=torch.uint8) for _ in range(12)]
        self.gpu_frames = True

    def emit(self, rgba):
        """The frame for the display: a GPU frame (interop) or bytes in PC memory."""
        if not self.gpu_frames:
            return self.download(rgba)
        buf = self.gring[self.gi]
        self.gi = (self.gi + 1) % len(self.gring)
        buf.copy_(rgba)
        ev = self.torch.cuda.Event()
        ev.record()
        return GpuFrame(buf, ev)

    def thread_init(self):
        """Called by the worker thread before using the engine."""
        self.ctx.__enter__()

    def set_options(self, sharpness, hud):
        if (sharpness, hud) != self.opts:
            self.sharp.fill_(float(sharpness) * SHARP_MAX)
            self.hud = bool(hud)
            self.opts = (sharpness, hud)

    def timestep(self, t):
        if t not in self.tsteps:
            sh, sw = int(self.ph * self.scale), int(self.pw * self.scale)
            self.tsteps[t] = self.torch.full((1, 1, sh, sw), t, device="cuda", dtype=self.dtype)
        return self.tsteps[t]

    def upload(self, frame):
        x = self.torch.from_numpy(frame).cuda(non_blocking=True).permute(2, 0, 1)[None]
        x = x.flip(1).to(self.dtype).div_(255)  # BGR -> RGB
        if self.resize:
            x = self.F.interpolate(x, size=(self.H, self.W), mode="bilinear", antialias=True, align_corners=False)
        return self.F.pad(x, (0, self.pw - self.W, 0, self.ph - self.H), mode="replicate")

    def _small(self, x):
        """Same reduced grayscale copy as small_gray() on the CPU, computed on the GPU (0-255 scale)."""
        rgb = x[:, :, :self.H, :self.W].float()
        g = (rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114) * 255
        return self.F.avg_pool2d(g, DUP_REDUCE)

    def small(self, x):
        return self.cache.get(x, "small", self._small)

    def is_dup(self, prev, cur):
        d = (self.small(cur) - self.small(prev)).abs()
        return (d >= DUP_LEVEL - 0.5).float().mean().item() < DUP_SHARE

    def tiny(self, x):
        return self.cache.get(x, "tiny", lambda h: self.F.adaptive_avg_pool2d(self.small(h), (18, 32))[0, 0]
                              .cpu().numpy())

    def finish(self, x, protect):
        """Crop, HUD protection (in-betweens only), sharpening: the frame as it will be shown, 0-1 RGB."""
        from post import apply_hud, sharpen
        x = x[:, :, :self.H, :self.W]
        if protect:  # the mask was computed once for this pair, in set_pair
            b = self.slot1[:, :, :self.H, :self.W]
            x = apply_hud(x, b, self.hudmask)
        return sharpen(x, self.sharp)

    def rgba_into(self, x, out, protect=False):
        """Padded RGB (0-1) -> RGBA bytes (alpha already 255), on the GPU."""
        y = self.finish(x, protect)[0]
        out[..., :3].copy_(y.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0))
        return out

    def download(self, rgba):
        buf = self.ring[self.ri]
        self.ri = (self.ri + 1) % len(self.ring)
        buf.copy_(rgba)
        return buf.numpy()

    def capture_graph(self, t):
        """Records a whole in-between frame (network + HUD protection + sharpening + RGBA conversion) as one
        CUDA graph: a single launch instead of a few hundred small ones."""
        torch = self.torch
        out = torch.full((self.H, self.W, 4), 255, device="cuda", dtype=torch.uint8)
        ts = self.timestep(t)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.rgba_into(self.net(self.slot0, self.slot1, ts, self.scale), out, True)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.rgba_into(self.net(self.slot0, self.slot1, ts, self.scale), out, True)
        self.graphs[t] = (g, out)

    def warmup(self):
        torch = self.torch
        with torch.inference_mode():
            x = torch.rand(1, 3, self.ph, self.pw, device="cuda", dtype=self.dtype)
            for _ in range(3):
                self.net(x, x, self.timestep(0.5), self.scale)
            if self.use_graphs:
                try:
                    for t in sorted({i / k for k in (2, 3, 4) for i in range(1, k)}):
                        self.capture_graph(t)
                except Exception as e:  # older driver or GPU: same result, a little slower
                    print(f"(CUDA graphs indisponibles, mode normal : {e})")
                    self.graphs.clear()
        torch.cuda.synchronize()

    def set_pair(self, prev, cur):
        from post import hud_count, hud_mask_small
        self.slot0.copy_(prev)
        self.slot1.copy_(cur)
        if self.hud:
            self.hudcount.copy_(hud_count(self.hudcount, self.small(prev), self.small(cur)))
            self.hudmask.copy_(hud_mask_small(self.hudcount))
        else:
            self.hudcount.zero_()
            self.hudmask.zero_()

    def reset_history(self):
        """After a scene cut: nothing on screen has been still for long."""
        self.hudcount.zero_()

    def interp(self, t):
        if t in self.graphs:
            g, out = self.graphs[t]
            g.replay()
            return self.emit(out)
        mid = self.net(self.slot0, self.slot1, self.timestep(t), self.scale)
        return self.emit(self.rgba_into(mid, self.rgba, True))

    def real(self, cur):
        return self.emit(self.rgba_into(cur, self.rgba))


def onnx_name(w, h, scale):
    return f"rife46_{w}x{h}_s{int(scale * 100)}.onnx"


def post_name(w, h):
    return f"post_{w}x{h}.onnx"


class DmlEngine:
    """The ONNX models take BGR frames as bytes and return RGBA bytes: colour conversion, padding, the
    network, HUD protection and sharpening all run on the graphics card (see tools/export_onnx.py)."""
    name = "DirectML"

    def __init__(self, src_size, work_size, scale):
        import onnxruntime as ort

        self.ort = ort
        self.W, self.H = work_size
        self.scale = scale
        self.resize = src_size != work_size
        models = app_dir() / "models"
        path = models / onnx_name(self.W, self.H, scale)
        if not path.exists():
            have = ", ".join(p.name for p in models.glob("rife*.onnx")) or "aucun"
            raise RuntimeError(f"Modèle {path.name} introuvable (modèles présents : {have}). "
                               "Utilise --res 1080p ou 1440p.")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = False  # required by DirectML
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(path), opts, providers=["DmlExecutionProvider"])
        if "DmlExecutionProvider" not in self.sess.get_providers():
            raise RuntimeError("DirectML indisponible sur cette machine.")
        self.post = ort.InferenceSession(str(models / post_name(self.W, self.H)), opts,
                                         providers=["DmlExecutionProvider"])
        self.pair = None
        self.tvals = {}
        self.cache = FrameCache()
        self.hudcount = None
        self.set_options(0.0, True)

    def thread_init(self):
        pass

    def set_options(self, sharpness, hud):
        self.sharp = np.array([float(sharpness) * SHARP_MAX], np.float32)
        self.hud = bool(hud)

    def upload(self, frame):
        if self.resize:
            frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame)

    def is_dup(self, prev, cur):
        return dup_cpu(self.cache.get(prev, "small", small_gray), self.cache.get(cur, "small", small_gray))

    def tiny(self, f):
        return self.cache.get(f, "tiny", lambda h: cv2.resize(self.cache.get(h, "small", small_gray), (32, 18),
                                                              interpolation=cv2.INTER_AREA).astype(np.float32))

    def warmup(self):
        f = np.zeros((self.H, self.W, 3), np.uint8)
        self.set_pair(f, f)
        for t in (0.5, 0.5, 0.5):
            self.interp(t)
        self.real(f)

    def set_pair(self, prev, cur):
        self.pair = (prev, cur)
        sa, sb = self.cache.get(prev, "small", small_gray), self.cache.get(cur, "small", small_gray)
        if self.hudcount is None or self.hudcount.shape != sa.shape:
            self.hudcount = np.zeros(sa.shape, np.float32)
        if self.hud:  # same rule as post.hud_count, on the CPU (the reduced copies are tiny)
            same = (cv2.absdiff(sa, sb) < HUD_LEVEL).astype(np.uint8)
            same = cv2.erode(same, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
            self.hudcount = (self.hudcount + 1) * same
        else:
            self.hudcount[:] = 0
        self.hudmask = (self.hudcount >= HUD_FRAMES).astype(np.float16)[None, None]

    def reset_history(self):
        if self.hudcount is not None:
            self.hudcount[:] = 0

    def interp(self, t):
        if t not in self.tvals:
            self.tvals[t] = np.array([t], np.float16)
        f0, f1 = self.pair
        return self.sess.run(None, {"frame0": f0, "frame1": f1, "t": self.tvals[t],
                                    "sharp": self.sharp, "hudmask": self.hudmask})[0]

    def real(self, cur):
        return self.post.run(None, {"frame": cur, "sharp": self.sharp})[0]