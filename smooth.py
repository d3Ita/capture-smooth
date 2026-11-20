"""Capture Smooth: reads a capture card (console, camera...), adds AI-generated in-between frames (RIFE),
shows the result full screen with low latency.

Keys while running:
  Échap      quitter
  Espace     interpolation on/off (pour comparer)
  2 / 3 / 4  multiplicateur d'images (x2, x3, x4)
  F1 ou I    afficher / masquer le compteur
"""
import argparse
import collections
import sys
import threading
import time

import cv2
import numpy as np

import glfw
from OpenGL.GL import (
    GL_BGRA, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_LINEAR, GL_ONE_MINUS_SRC_ALPHA, GL_QUADS, GL_RGBA, GL_RGBA8,
    GL_SRC_ALPHA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE, glBegin, glBindTexture, glBlendFunc, glClear, glClearColor, glDisable, glEnable, glEnd,
    glGenTextures, glPixelStorei, glTexCoord2f, glTexImage2D, glTexParameteri, glTexSubImage2D, glVertex2f,
    glViewport,
)

APP = "Capture Smooth"
VERSION = "1.1.0"

RES = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160)}
now = time.perf_counter


def sleep_until(t):
    while True:
        d = t - now()
        if d <= 0:
            return
        time.sleep(d - 0.0015 if d > 0.002 else 0)


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.inp = self.dups = self.out = self.dropped = 0
        self.ai_ms = 0.0
        self.ai_n = 0
        self.lat = []

    def snapshot(self):
        with self.lock:
            s = (self.inp, self.dups, self.out, self.dropped, self.ai_ms / max(self.ai_n, 1),
                 sum(self.lat) / len(self.lat) if self.lat else 0.0)
            self.reset()
        return s


class State:
    def __init__(self, mult):
        self.on = True
        self.mult = mult
        self.running = True


# ---------------------------------------------------------------- sources

class Source(threading.Thread):
    """Keeps only the newest frame; consumers wait for a new sequence number."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cond = threading.Condition()
        self.frame = None
        self.t = 0.0
        self.seq = 0
        self.running = True
        self.error = None

    def publish(self, frame):
        with self.cond:
            self.frame, self.t, self.seq = frame, now(), self.seq + 1
            self.cond.notify_all()

    def wait_new(self, last_seq, timeout):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout)
            if self.seq == last_seq:
                return None
            return self.frame, self.t, self.seq


class CaptureSource(Source):
    def __init__(self, index, w, h, fps):
        super().__init__()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir le boîtier de capture. Ferme OBS et le logiciel du boîtier (ils le bloquent) puis relance.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Le boîtier s'ouvre mais n'envoie pas d'image. Vérifie que la console est allumée et branchée.")
        self.size = (frame.shape[1], frame.shape[0])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def run(self):
        fails = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 120:
                    self.error = "Le flux du boîtier s'est arrêté."
                    break
                time.sleep(0.005)
                continue
            fails = 0
            self.publish(frame)
        self.cap.release()


class TestSource(Source):
    """Synthetic moving picture, to try the app without the console."""

    def __init__(self, w, h, fps, game_fps):
        super().__init__()
        self.size, self.fps, self.game_fps = (w, h), fps, game_fps
        self.bg = np.zeros((h, w, 3), np.uint8)
        for x in range(0, w, 160):
            cv2.line(self.bg, (x, 0), (x, h), (40, 40, 40), 2)

    def draw(self, t):
        w, h = self.size
        f = self.bg.copy()
        x = int((t * 900) % (w + 200)) - 100
        cv2.circle(f, (x, h // 2), 90, (60, 180, 255), -1)
        cv2.rectangle(f, (w - x - 60, h // 4 - 60), (w - x + 60, h // 4 + 60), (255, 120, 60), -1)
        y = int((np.sin(t * 2.5) * 0.4 + 0.5) * h)
        cv2.rectangle(f, (0, y - 6), (w, y + 6), (200, 200, 200), -1)
        cv2.putText(f, f"TEST {t:7.2f}s", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4)
        return f

    def run(self):
        t0 = now()
        n = 0
        game_t = None
        frame = None
        while self.running:
            n += 1
            sleep_until(t0 + n / self.fps)
            t = now() - t0
            # A 30 fps game in a 60 Hz signal: every other frame is a copy.
            gt = int(t * self.game_fps) / self.game_fps
            if gt != game_t:
                game_t, frame = gt, self.draw(gt)
            self.publish(frame)


# ---------------------------------------------------------------- AI worker

class Worker(threading.Thread):
    """Feeds captured frames to the AI engine and schedules in-betweens and real frames on the display.
    Frames reach the display as RGBA bytes (the fastest texture upload by far)."""

    def __init__(self, src, display, state, stats, engine):
        super().__init__(daemon=True)
        self.src, self.display, self.state, self.stats, self.engine = src, display, state, stats, engine
        self.error = None

    def run(self):
        try:
            self.engine.thread_init()
            self.loop()
        except Exception as e:  # surface GPU errors to the main thread
            self.error = repr(e)
            self.state.running = False

    def loop(self):
        eng = self.engine
        last_seq = 0
        prev = None
        period = 1.0 / max(self.src.fps or 60, 1)
        last_t = None
        while self.state.running:
            item = self.src.wait_new(last_seq, 0.2)
            if item is None:
                continue
            frame, t_in, last_seq = item
            cur = eng.upload(frame)

            if prev is not None and eng.is_dup(prev, cur):
                with self.stats.lock:
                    self.stats.dups += 1
                continue
            if last_t is not None:
                dt = t_in - last_t
                if dt < 2.5 * period:
                    period = min(max(0.85 * period + 0.15 * dt, 1 / 240), 1 / 20)
            last_t = t_in
            with self.stats.lock:
                self.stats.inp += 1

            k = self.state.mult if self.state.on else 1
            if prev is None or k == 1:
                self.display.push(eng.real(cur), now(), t_in)
                prev = cur
                continue

            # In-betweens first: the real frame is only shown after them, so it can wait.
            eng.set_pair(prev, cur)
            base = None
            next_expected = t_in + period
            for i in range(1, k):
                if i > 1 and now() > next_expected:
                    break  # the next game frame is due: drop the remaining in-betweens to keep latency down
                t0 = now()
                img = eng.interp(i / k)
                done = now()
                with self.stats.lock:
                    self.stats.ai_ms += (done - t0) * 1000
                    self.stats.ai_n += 1
                if base is None:
                    base = done
                self.display.push(img, base + (i - 1) * period / k, None)
            self.display.push(eng.real(cur), max(now(), base + (k - 1) * period / k), t_in)
            prev = cur


# ---------------------------------------------------------------- display

class Display:
    def __init__(self, size, mode_name, vsync, state, stats):
        self.W, self.H = size
        self.state, self.stats = state, stats
        self.queue = collections.deque()
        self.cond = threading.Condition()

        if not glfw.init():
            raise RuntimeError("OpenGL indisponible")
        mon = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(mon)
        if mode_name == "fenetre":
            self.win = glfw.create_window(1280, 720, APP, None, None)
        elif mode_name == "sans-bordure":
            # Borderless window covering the screen, minus one pixel of (black) width. A window that covers
            # the monitor exactly is treated by Windows as fullscreen: the screen re-syncs (black flash) each
            # time focus moves to the other monitor. One pixel short keeps it a normal window; the taskbar,
            # which Windows only hides for real fullscreen windows, is hidden by taskbar.py instead.
            glfw.window_hint(glfw.DECORATED, False)
            mx, my = glfw.get_monitor_pos(mon)
            self.win = glfw.create_window(mode.size.width - 1, mode.size.height, APP, None, None)
            glfw.set_window_pos(self.win, mx, my)
            from taskbar import hide_taskbar
            hide_taskbar()
        else:
            glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)
            self.win = glfw.create_window(mode.size.width, mode.size.height, APP, mon, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(1 if vsync else 0)
        glfw.set_input_mode(self.win, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        glfw.set_key_callback(self.win, self.on_key)
        self.refresh = mode.refresh_rate

        glClearColor(0, 0, 0, 1)
        glEnable(GL_TEXTURE_2D)
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.W, self.H, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

        # Small on-screen counter (top-left), toggled with F1 or I.
        self.snapshot = None  # (path, time): save one displayed frame, for checking the app
        self.audio = None
        self.overlay_on = True
        self.overlay_info = None
        self.overlay_ver = 0
        self.overlay_drawn = -1
        self.overlay_size = (0, 0)
        self.otex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.otex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, self.tex)

    def set_info(self, info):
        """Called once per second by the stats thread; drawn by the display thread."""
        self.overlay_info = info
        self.overlay_ver += 1

    def render_overlay(self):
        fps, line2, line3 = self.overlay_info
        w, h = 430, 98
        img = np.zeros((h, w, 4), np.uint8)
        img[:] = (0, 0, 0, 150)
        color = (120, 230, 120, 255) if fps >= 100 else (80, 210, 255, 255) if fps >= 55 else (90, 90, 255, 255)
        cv2.putText(img, f"{fps:3d} FPS", (12, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
        cv2.putText(img, line2, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235, 255), 1, cv2.LINE_AA)
        cv2.putText(img, line3, (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170, 255), 1, cv2.LINE_AA)
        glBindTexture(GL_TEXTURE_2D, self.otex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, img)
        self.overlay_size = (w, h)

    def draw_quad(self):
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, 1)
        glTexCoord2f(0, 0); glVertex2f(-1, 1)
        glEnd()

    def draw_overlay(self, ix, iy, vw, vh):
        if not self.overlay_on or self.overlay_info is None:
            return
        ver = self.overlay_ver
        if ver != self.overlay_drawn:
            self.render_overlay()
            self.overlay_drawn = ver
        ow, oh = self.overlay_size
        glViewport(ix + 16, iy + vh - oh - 16, ow, oh)
        glBindTexture(GL_TEXTURE_2D, self.otex)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.draw_quad()
        glDisable(GL_BLEND)
        glBindTexture(GL_TEXTURE_2D, self.tex)

    def on_key(self, win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key in (glfw.KEY_F1, glfw.KEY_I):
            self.overlay_on = not self.overlay_on
        elif key == glfw.KEY_SEMICOLON or key == glfw.KEY_M:  # M sits on ';' with an AZERTY keyboard
            if self.audio:
                self.audio.muted = not self.audio.muted
                print(f"\nSon {'coupé' if self.audio.muted else 'activé'}")
        elif key == glfw.KEY_SPACE:
            self.state.on = not self.state.on
            print(f"\nInterpolation {'ACTIVÉE' if self.state.on else 'DÉSACTIVÉE'}")
        elif key in (glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_KP_2, glfw.KEY_KP_3, glfw.KEY_KP_4):
            self.state.mult = {glfw.KEY_2: 2, glfw.KEY_3: 3, glfw.KEY_4: 4,
                               glfw.KEY_KP_2: 2, glfw.KEY_KP_3: 3, glfw.KEY_KP_4: 4}[key]
            print(f"\nMultiplicateur x{self.state.mult}")

    def push(self, img, target, t_in):
        with self.cond:
            self.queue.append((target, img, t_in))
            self.cond.notify()

    def next_job(self):
        with self.cond:
            if not self.queue:
                self.cond.wait(0.005)
            if not self.queue:
                return None
            # Frames already late are skipped, except the newest one, so we never fall behind.
            t = now()
            while len(self.queue) > 1 and self.queue[1][0] <= t:
                self.queue.popleft()
                with self.stats.lock:
                    self.stats.dropped += 1
            return self.queue.popleft()

    def present(self, img):
        fw, fh = glfw.get_framebuffer_size(self.win)
        glViewport(0, 0, fw, fh)
        glClear(GL_COLOR_BUFFER_BIT)
        s = min(fw / self.W, fh / self.H)  # keep 16:9, black bars on an ultrawide screen
        vw, vh = int(self.W * s), int(self.H * s)
        ix, iy = (fw - vw) // 2, (fh - vh) // 2
        glViewport(ix, iy, vw, vh)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.W, self.H, GL_RGBA, GL_UNSIGNED_BYTE, img)  # RGBA: 5x faster than BGR
        self.draw_quad()
        self.draw_overlay(ix, iy, vw, vh)
        if self.snapshot and now() >= self.snapshot[1]:
            from OpenGL.GL import GL_RGB, glReadPixels
            raw = glReadPixels(0, 0, fw, fh, GL_RGB, GL_UNSIGNED_BYTE)
            shot = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3)[::-1, :, ::-1]
            cv2.imwrite(self.snapshot[0], shot)
            self.snapshot = None
        glfw.swap_buffers(self.win)

    def loop(self):
        while not glfw.window_should_close(self.win) and self.state.running:
            glfw.poll_events()
            job = self.next_job()
            if job is None:
                continue
            target, img, t_in = job
            sleep_until(target)
            self.present(img)
            shown = now()
            with self.stats.lock:
                self.stats.out += 1
                if t_in is not None:
                    self.stats.lat.append((shown - t_in) * 1000)
        glfw.terminate()
        from taskbar import show_taskbar
        show_taskbar()


# ---------------------------------------------------------------- main

VIRTUAL = ("virtual", "lsvcam", "screen link", "droidcam", "broadcast", "snap camera", "iriun", "xsplit")
CAPTURE_HINTS = ("capture", "hd60", "4k60", "4k x", "4k s", "elgato", "cam link", "avermedia", "live gamer",
                 "razer ripsaw", "usb3", "hdmi", "game")


def pick_device(wanted):
    """Index and name of the capture device: by number, by part of its name, or "auto"."""
    if wanted.isdigit():
        return int(wanted), f"appareil {wanted}"
    try:
        from devices import video_devices
        names = video_devices()
    except Exception as e:
        print(f"(liste des appareils indisponible : {e})")
        return 0, "appareil 0"
    if wanted.lower() != "auto":
        for i, n in enumerate(names):
            if wanted.lower() in n.lower():
                return i, n
        raise RuntimeError(f"Aucun appareil « {wanted} » trouvé. Appareils vus : {', '.join(names) or 'aucun'}")
    real = [(i, n) for i, n in enumerate(names) if not any(v in n.lower() for v in VIRTUAL)]
    for i, n in real:
        if any(c in n.lower() for c in CAPTURE_HINTS):
            return i, n
    if real:
        return real[0]
    raise RuntimeError(f"Aucun boîtier de capture trouvé. Appareils vus : {', '.join(names) or 'aucun'}. "
                       "Choisis-en un avec --device \"nom\".")


def make_engine(kind, src_size, work, scale):
    """CUDA if this card supports it, otherwise DirectML. Each engine is warmed up here, so a card that
    loads but cannot run the network still falls back instead of crashing later."""
    reasons = []
    if kind == "auto":
        try:
            import torch  # noqa: F401  (absent from the DirectML build: go straight to DirectML, silently)
        except ImportError:
            kind = "directml"
    if kind in ("auto", "cuda"):
        try:
            from engines import CudaEngine
            eng = CudaEngine(src_size, work, scale)
            eng.warmup()
            if getattr(eng, "dtype", None) is not None and "float32" in str(eng.dtype):
                print(f"({eng.gpu} : calcul en pleine précision, plus adapté à cette carte)")
            return eng
        except Exception as e:
            reasons.append(f"CUDA : {e}")
            if kind == "cuda":
                raise RuntimeError(f"moteur CUDA indisponible ({e})")
            print(f"Moteur CUDA indisponible ({e}) : passage au moteur DirectML.")
    try:
        from engines import DmlEngine
        eng = DmlEngine(src_size, work, scale)
        eng.warmup()
        return eng
    except Exception as e:
        reasons.append(f"DirectML : {e}")
    raise RuntimeError("aucun moteur IA ne marche sur cette carte graphique.\n  " + "\n  ".join(reasons) +
                       "\n  Mets à jour le pilote de ta carte graphique (Windows 10/11 et DirectX 12 requis).")


def main():
    ap = argparse.ArgumentParser(description=f"{APP} : boîtier de capture + images intermédiaires par IA")
    ap.add_argument("--res", choices=RES, default="1440p", help="résolution de capture (défaut 1440p)")
    ap.add_argument("--fps", type=int, default=60, help="fps de capture (défaut 60)")
    ap.add_argument("--x", type=int, choices=[2, 3, 4], default=2, help="multiplicateur d'images (défaut x2)")
    ap.add_argument("--device", default="auto", help="nom (ou numéro) du boîtier de capture (défaut : auto)")
    ap.add_argument("--moteur", choices=["auto", "cuda", "directml"], default="auto",
                    help="cuda = NVIDIA (le plus rapide), directml = toute carte DirectX 12 ; auto = cuda si possible")
    ap.add_argument("--list", action="store_true", help="affiche les appareils vidéo et quitte")
    ap.add_argument("--show-taskbar", action="store_true",
                    help="réaffiche la barre des tâches (si l'appli a été fermée de force) et quitte")
    ap.add_argument("--quality", choices=["rapide", "normal"], default="rapide",
                    help="rapide = IA à demi-résolution (conseillé en 1440p)")
    ap.add_argument("--mode", choices=["sans-bordure", "plein-ecran", "fenetre"], default="sans-bordure",
                    help="sans-bordure (défaut) = fenêtre qui couvre l'écran ; plein-ecran = plein écran exclusif ; "
                         "fenetre = petite fenêtre")
    ap.add_argument("--windowed", action="store_true", help="raccourci pour --mode fenetre")
    ap.add_argument("--vsync", action="store_true", help="active la V-Sync (plus de latence, pas de déchirure)")
    ap.add_argument("--test", action="store_true", help="image de test animée au lieu du boîtier")
    ap.add_argument("--test-game-fps", type=int, default=30, help="fps du jeu simulé en mode test")
    ap.add_argument("--seconds", type=float, default=0, help="quitte tout seul après N secondes (pour tester)")
    ap.add_argument("--snapshot", default="", help=argparse.SUPPRESS)  # save one displayed frame after 5 s
    ap.add_argument("--no-audio", action="store_true", help="ne pas jouer le son du boîtier")
    ap.add_argument("--audio-delay", type=float, default=0, help="retarde le son de N ms si il arrive avant l'image")
    ap.add_argument("--hide-counter", action="store_true", help="démarre avec le compteur masqué (F1 l'affiche)")
    if len(sys.argv) == 1:  # double-click on the .exe: settings window
        from launcher import choose
        chosen = choose(APP)
        if chosen is None:
            return
        args = ap.parse_args(chosen)
    else:
        args = ap.parse_args()

    if args.show_taskbar:
        from taskbar import show_taskbar
        show_taskbar()
        print("Barre des tâches réaffichée.")
        return
    if args.list:
        from devices import video_devices
        for i, n in enumerate(video_devices()):
            print(f"{i}: {n}")
        return

    w, h = RES[args.res]
    if args.test:
        src = TestSource(w, h, args.fps, args.test_game_fps)
        print(f"Source : image de test {w}x{h} à {args.fps} Hz (jeu simulé à {args.test_game_fps} fps)")
    else:
        idx, name = pick_device(args.device)
        src = CaptureSource(idx, w, h, args.fps)
        print(f"Source : {name} en {src.size[0]}x{src.size[1]} à {src.fps:.0f} fps")
        if src.size != (w, h):
            print(f"  (le boîtier a refusé {w}x{h} : vérifie la résolution de sortie de la console)")

    sw, sh = src.size
    work = (sw, sh) if sh <= 1440 else (round(sw * 1440 / sh), 1440)
    scale = 0.5 if (args.quality == "rapide" and work[1] >= 1440) else 1.0

    print(f"{APP} {VERSION} — préparation de l'IA…")
    engine = make_engine(args.moteur, src.size, work, scale)
    state, stats = State(args.x), Stats()
    display = Display(work, "fenetre" if args.windowed else args.mode, args.vsync, state, stats)
    worker = Worker(src, display, state, stats, engine)
    print(f"Prêt. Moteur {engine.name}, image {work[0]}x{work[1]}, IA échelle {scale}, x{args.x}, "
          f"écran {display.refresh} Hz.")
    audio = None
    if not args.test and not args.no_audio:
        try:
            from audio import AudioPassthrough
            audio = AudioPassthrough(wanted=name, delay_ms=args.audio_delay)
            audio.start()
            lat = sum(audio.stream.latency) * 1000
            print(f"Son : {audio.in_name} -> {audio.out_name} ({audio.api}, ~{lat:.0f} ms)")
        except Exception as e:
            print(f"Son : désactivé ({e})")
    display.audio = audio
    display.overlay_on = not args.hide_counter
    print("Échap = quitter · Espace = interpolation on/off · 2/3/4 = multiplicateur · F1 ou I = compteur · M = son\n")

    def report():
        while state.running:
            time.sleep(1)
            inp, dups, out, dropped, ai, lat = stats.snapshot()
            line = (f"\rentrée {inp:3d} img/s (doublons {dups:3d}) | sortie {out:3d} img/s | "
                    f"IA {ai:4.1f} ms | retard ajouté ~{lat:4.1f} ms | sautées {dropped:3d}   ")
            print(line, end="", flush=True)
            # Text for the on-screen counter (plain ASCII: the OpenCV font has no accents).
            mode = f"x{state.mult}" if state.on else "interpolation OFF"
            moving = f"jeu {inp} fps" if inp else "image fixe"
            display.set_info((
                out,
                f"{moving}  {mode}  IA {ai:.0f} ms  retard +{lat:.0f} ms",
                f"{work[0]}x{work[1]}  IA {int(scale * 100)}%  son "
                f"{'OFF' if not display.audio or display.audio.muted else 'ON'}  (F1 = masquer)",
            ))
            if src.error:
                print("\n" + src.error)
                state.running = False

    src.start()
    worker.start()
    threading.Thread(target=report, daemon=True).start()
    if args.snapshot:
        display.snapshot = (args.snapshot, now() + 5)
    if args.seconds:
        threading.Timer(args.seconds, lambda: glfw.set_window_should_close(display.win, True)).start()
    try:
        display.loop()
    finally:
        state.running = False
        src.running = False
        with src.cond:
            src.cond.notify_all()
        worker.join(timeout=2)
        src.join(timeout=2)
        if audio:
            audio.stop()
    if worker.error:
        print("\nErreur IA :", worker.error)
    print("\nTerminé.")


def pause_if_double_clicked():
    """A double-clicked .exe closes its window at once: keep it open so the message can be read."""
    if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
        try:
            input("\nAppuie sur Entrée pour fermer.")
        except EOFError:
            pass


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"Erreur : {e}")
        pause_if_double_clicked()
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        pause_if_double_clicked()
        sys.exit(1)
