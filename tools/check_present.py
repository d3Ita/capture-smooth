"""Off-screen check of the display shader's picture effects (present.py): debanding, local contrast, vibrance.

Renders test pictures through the real shader into an off-screen buffer, with and without the effects, and checks:
  - a banded gradient (steps of 2 levels) comes out smooth;
  - a sharp edge and a 1-pixel line stay sharp, with no visible halo;
  - already saturated colours (and a skin tone) barely move, while a dull colour gains saturation.
Then measures the cost at 1440p with OpenGL timer queries (mip chain refresh + draw, as in the app).

    python tools/check_present.py                     checks + cost
    python tools/check_present.py --image a.png ...   also renders those pictures with and without the effects
                                                      (side-by-side crops written to --out)
Exit code 1 if a check fails.
"""
import argparse
import ctypes
import sys
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL.GL import (
    GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_QUADS,
    GL_RGB, GL_RGBA, GL_RGBA8, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, GL_TIME_ELAPSED, GL_UNPACK_ALIGNMENT, GL_PACK_ALIGNMENT, GL_UNSIGNED_BYTE,
    glBegin, glBeginQuery, glBindFramebuffer, glBindTexture, glCheckFramebufferStatus, glDeleteTextures, glEnable,
    glEnd, glEndQuery, glFinish, glFramebufferTexture2D, glGenFramebuffers, glGenQueries, glGenTextures,
    glGenerateMipmap, glPixelStorei, glReadPixels, glTexCoord2f, glTexImage2D, glTexParameteri, glTexStorage2D,
    glTexSubImage2D, glVertex2f, glViewport, GL_QUERY_RESULT,
)
from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from present import FX_DEFAULTS, Presenter  # noqa: E402


class Renderer:
    """A hidden OpenGL window, a source texture set up like the app's (full mip chain) and an off-screen target."""

    def __init__(self):
        if not glfw.init():
            raise RuntimeError("OpenGL indisponible")
        glfw.window_hint(glfw.VISIBLE, False)
        self.win = glfw.create_window(64, 64, "check_present", None, None)
        glfw.make_context_current(self.win)
        glEnable(GL_TEXTURE_2D)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        self.fx_prog = Presenter(effects=True)
        if not self.fx_prog.fx:
            raise RuntimeError(f"le shader des effets ne compile pas : {self.fx_prog.fx_error}")
        self.plain_prog = Presenter(effects=False)
        self.src = self.dst = None
        self.fbo = glGenFramebuffers(1)
        self.query = glGenQueries(1)[0] if hasattr(glGenQueries(1), "__len__") else glGenQueries(1)

    def set_source(self, rgb):
        h, w = rgb.shape[:2]
        if self.src is None or self.src[1:] != (w, h):
            if self.src is not None:
                glDeleteTextures([self.src[0]])
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            for p, v in ((GL_TEXTURE_MAG_FILTER, GL_LINEAR), (GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                         (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
                glTexParameteri(GL_TEXTURE_2D, p, v)
            glTexStorage2D(GL_TEXTURE_2D, int(np.floor(np.log2(max(w, h)))) + 1, GL_RGBA8, w, h)
            self.src = (tex, w, h)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        glBindTexture(GL_TEXTURE_2D, self.src[0])
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, np.ascontiguousarray(rgba))

    def set_target(self, w, h):
        if self.dst is None or self.dst[1:] != (w, h):
            if self.dst is not None:
                glDeleteTextures([self.dst[0]])
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("tampon hors écran incomplet")
            self.dst = (tex, w, h)

    def draw(self, fx, view=None, sharp=True):
        """One frame as the app draws it: mip refresh when local contrast is on, then the shader pass."""
        tex, sw, sh = self.src
        vw, vh = view or (sw, sh)
        self.set_target(vw, vh)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, vw, vh)
        glBindTexture(GL_TEXTURE_2D, tex)
        prog = self.fx_prog if fx is not None else self.plain_prog
        lod = -1.0
        if fx and fx.get("lcon", 0) > 0:
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            lod = prog.blur_lod(sw, vw)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        prog.begin(sw, sh, vw, vh, sharp, False, fx, lod)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, 1)
        glTexCoord2f(0, 0); glVertex2f(-1, 1)
        glEnd()
        prog.end()

    def render(self, rgb, fx, view=None):
        self.set_source(rgb)
        self.draw(fx, view)
        vw, vh = self.dst[1:]
        raw = glReadPixels(0, 0, vw, vh, GL_RGB, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return np.frombuffer(raw, np.uint8).reshape(vh, vw, 3)[::-1].copy()

    def cost_ms(self, rgb, fx, n=300):
        """Mean GPU time of one frame (mip refresh + draw), OpenGL timer queries."""
        self.set_source(rgb)
        for _ in range(20):
            self.draw(fx)
        glFinish()
        ns, total = ctypes.c_uint64(0), 0
        for _ in range(n):
            glBeginQuery(GL_TIME_ELAPSED, self.query)
            self.draw(fx)
            glEndQuery(GL_TIME_ELAPSED)
            glGetQueryObjectui64v(self.query, GL_QUERY_RESULT, ctypes.byref(ns))
            total += ns.value
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return total / n / 1e6


# ------------------------------------------------------------------ test pictures and checks

def banded(w=1280, h=720):
    """A dark sky-like gradient quantized in 16 bands of 2 levels: the banding the effect must remove."""
    x = np.arange(w)
    row = (40 + 2 * (x * 16 // w)).astype(np.uint8)
    return np.repeat(np.repeat(row[None, :, None], h, 0), 3, 2)


def edges(w=1280, h=720):
    img = np.full((h, w, 3), 30, np.uint8)
    img[:, w // 2:] = 220             # hard edge, dark to bright
    img[:, w // 4] = 230              # 1-pixel bright line on dark
    return img


PATCHES = {  # name: RGB
    "rouge sature": (230, 30, 30), "vert sature": (30, 200, 60), "bleu sature": (40, 60, 230),
    "peau": (224, 172, 140), "terne": (92, 106, 126), "gris": (128, 128, 128),  # terne: dull blue-grey (a dull
    # brown would count as a skin tone, which vibrance leaves alone on purpose)
}


def patches(size=96):
    img = np.zeros((size, size * len(PATCHES), 3), np.uint8)
    for i, rgb in enumerate(PATCHES.values()):
        img[:, i * size:(i + 1) * size] = rgb
    return img


def saturation(rgb):
    rgb = np.asarray(rgb, np.float64)
    return rgb.max() - rgb.min()


def check(r, fx):
    ok = True

    def verdict(name, good, text):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'OK' if good else 'ECHEC'}] {name} : {text}")

    # 1. Banding: average over rows (the dither is noise, its row mean is ~0), count steps of 1 level or more.
    img = banded()
    for label, out in (("sans", r.render(img, None)), ("avec", r.render(img, fx))):
        prof = out[:, :, 1].astype(np.float64).mean(0)[40:-40]  # away from the picture's own borders
        jumps = np.abs(np.diff(prof))
        if label == "sans":
            before = (int((jumps >= 1.0).sum()), float(jumps.max()))
        else:
            after = (int((jumps >= 1.0).sum()), float(jumps.max()))
    verdict("degrade a bandes lisse", after[0] <= 1 and after[1] < 1.0,
            f"marches de 1 niveau ou plus {before[0]} -> {after[0]}, plus grande marche {before[1]:.2f} -> "
            f"{after[1]:.2f} niveau")

    # 2. Sharp edge and thin line.
    img = edges()
    a, b = r.render(img, None).astype(np.float64), r.render(img, fx).astype(np.float64)
    pa, pb = a[:, :, 1].mean(0), b[:, :, 1].mean(0)
    w = img.shape[1]

    def width(p):  # 10-90 % rise around the hard edge, in pixels
        seg = p[w // 2 - 40:w // 2 + 40]
        lo, hi = seg.min(), seg.max()
        return int(((seg > lo + 0.1 * (hi - lo)) & (seg < lo + 0.9 * (hi - lo))).sum())

    over = max(pb[w // 2:w // 2 + 40].max() - pa[w // 2:w // 2 + 40].max(),
               pa[w // 2 - 40:w // 2].min() - pb[w // 2 - 40:w // 2].min())
    line_a, line_b = pa[w // 4] - pa[w // 4 - 8], pb[w // 4] - pb[w // 4 - 8]
    verdict("contour net reste net", width(pb) <= width(pa) and over <= 4.0,
            f"largeur de la transition {width(pa)} -> {width(pb)} px, halo {over:.1f} niveaux")
    verdict("trait de 1 pixel garde", line_b >= 0.97 * line_a,
            f"contraste du trait {line_a:.0f} -> {line_b:.0f} niveaux")

    # 3. Colours: mean of each patch, before and after.
    img, size = patches(), 96
    a, b = r.render(img, None).astype(np.float64), r.render(img, fx).astype(np.float64)
    moves = {}
    for i, name in enumerate(PATCHES):
        ma = a[16:-16, i * size + 16:(i + 1) * size - 16].reshape(-1, 3).mean(0)
        mb = b[16:-16, i * size + 16:(i + 1) * size - 16].reshape(-1, 3).mean(0)
        moves[name] = (float(np.abs(mb - ma).max()), saturation(ma), saturation(mb))
    sat_move = max(moves[n][0] for n in ("rouge sature", "vert sature", "bleu sature"))
    verdict("couleurs saturees a peine changees", sat_move <= 3.0, f"au plus {sat_move:.1f} niveaux")
    verdict("ton chair garde", moves["peau"][0] <= 3.0, f"{moves['peau'][0]:.1f} niveaux")
    verdict("couleur terne ravivee", moves["terne"][2] > moves["terne"][1] + 2.0,
            f"saturation {moves['terne'][1]:.0f} -> {moves['terne'][2]:.0f}")
    verdict("gris inchange", moves["gris"][0] <= 1.0, f"{moves['gris'][0]:.1f} niveau")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="*", default=[], help="images (PNG/JPG) à rendre avec et sans effets")
    ap.add_argument("--out", default=".", help="dossier des comparaisons")
    ap.add_argument("--debanding", type=float, default=FX_DEFAULTS["deband"])
    ap.add_argument("--contraste-local", type=float, default=FX_DEFAULTS["lcon"])
    ap.add_argument("--vibrance", type=float, default=FX_DEFAULTS["vib"])
    args = ap.parse_args()
    fx = {"deband": args.debanding, "lcon": args.contraste_local, "vib": args.vibrance}

    r = Renderer()
    print(f"Effets : debanding {fx['deband']}, contraste local {fx['lcon']}, vibrance {fx['vib']}")
    ok = check(r, fx)

    print("Cout GPU par image en 2560x1440 (rafraichissement des mipmaps + passage du shader) :")
    test = cv2.resize(patches(), (2560, 1440), interpolation=cv2.INTER_NEAREST)
    base = r.cost_ms(test, None)
    configs = [("shader actuel (sans effets)", None), ("les trois effets", fx),
               ("debanding seul", {"deband": fx["deband"]}), ("contraste local seul", {"lcon": fx["lcon"]}),
               ("vibrance seule", {"vib": fx["vib"]})]
    for name, cfg in configs:
        ms = base if cfg is None else r.cost_ms(test, cfg)
        extra = "" if cfg is None else f"  (+{ms - base:.3f} ms)"
        print(f"  {name:30s} {ms:.3f} ms{extra}")
    total = r.cost_ms(test, fx) - base
    good = total < 0.5
    ok &= good
    print(f"  [{'OK' if good else 'ECHEC'}] cout des effets sous 0,5 ms : {total:.3f} ms")

    out = Path(args.out)
    for path in args.image:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  (image illisible : {path})")
            continue
        rgb = bgr[:, :, ::-1].copy()
        a, b = r.render(rgb, None), r.render(rgb, fx)
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        print(f"  {Path(path).name} : ecart moyen {diff.mean():.2f} niveau, 99e centile {np.percentile(diff, 99):.0f}")
        stem = out / Path(path).stem
        cv2.imwrite(str(stem) + "_sans.png", a[:, :, ::-1])
        cv2.imwrite(str(stem) + "_avec.png", b[:, :, ::-1])
    glfw.terminate()
    print("Tout est bon." if ok else "Au moins une verification a echoue.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
