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
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

import glfw
from OpenGL.GL import (
    GL_BGRA, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_LINEAR, GL_ONE_MINUS_SRC_ALPHA, GL_QUADS, GL_RGBA, GL_RGBA8,
    GL_CLAMP_TO_EDGE, GL_SRC_ALPHA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE, glBegin, glBindTexture, glBlendFunc, glClear, glClearColor, glDisable, glEnable, glEnd,
    glGenTextures, glPixelStorei, glTexCoord2f, glTexImage2D, glTexParameteri, glTexSubImage2D, glVertex2f,
    glViewport,
)

APP = "Capture Smooth"
VERSION = "1.3.0"

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
        self.inp = self.dups = self.out = self.dropped = self.cuts = 0
        self.ai_ms = 0.0
        self.ai_n = 0
        self.lat = []

    def snapshot(self):
        with self.lock:
            s = (self.inp, self.dups, self.out, self.dropped, self.ai_ms / max(self.ai_n, 1),
                 sum(self.lat) / len(self.lat) if self.lat else 0.0, self.cuts)
            self.reset()
        return s


class State:
    def __init__(self, mult, sharpness=0.3, hud=True, cuts=True):
        self.on = True
        self.mult = mult
        self.mult_eff = mult  # what the card actually keeps up with (automatic multiplier)
        self.running = True
        self.sharpness = sharpness      # 0-1, the amount chosen in the settings
        self.sharp_on = sharpness > 0   # S key
        self.hud = hud                  # H key
        self.cuts = cuts


# Scene cuts (camera cut, cutscene, respawn): an in-between of two unrelated pictures is a ghostly blend,
# so the new picture is shown directly. Measured on GTA 6 driving footage with fast turns: on a 32x18
# thumbnail, motion never jumped above 12 levels AND 3x the recent average at once (0 / 1952 pairs),
# while 98.7% of cuts did.
CUT_LEVEL = 12.0
CUT_RATIO = 3.0


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


# Programs that keep a capture card to themselves while they run.
BLOCKERS = {"obs64": "OBS", "obs32": "OBS", "elgato studio": "Elgato Studio", "4kcaptureutility": "Elgato 4K Capture Utility",
            "gamecapture": "Elgato Game Capture", "streamlabs obs": "Streamlabs", "streamlabs desktop": "Streamlabs",
            "windowscamera": "l'appli Caméra de Windows", "xsplit.core": "XSplit", "recentral": "AVerMedia RECentral"}
# Discord, Teams or Zoom are left out: they are often open without using the camera, and a wrong lead is
# worse than none.


def capture_blockers():
    """Names of running programs that commonly hold a capture card."""
    try:
        import subprocess
        out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=5,
                             creationflags=0x08000000).stdout.lower()  # no console window flash
    except Exception:
        return []
    found = []
    for exe, label in BLOCKERS.items():
        if f'"{exe}.exe"' in out and label not in found:
            found.append(label)
    return found


class CaptureSource(Source):
    # Uncompressed formats first. OpenCV already prefers YUY2 over MJPG (a JPEG per frame: blocky, smeared
    # colour), but ranks NV12 after MJPG, so cards offering only NV12 uncompressed got MJPG. A card that cannot
    # carry the asked size and fps uncompressed (4K60 on USB 3.0) gets MJPG, then its default.
    FORMATS = ("YUY2", "NV12", "MJPG", None)  # MJPG at full fps beats YUY2 at half fps
    UNCOMPRESSED = ("YUY2", "NV12")

    @staticmethod
    def fourcc_name(cap):
        v = int(cap.get(cv2.CAP_PROP_FOURCC))
        name = "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\0 ")
        return name if name.isprintable() and name else "?"

    def open_format(self, index, w, h, fps, fmt):
        """The card opened in this format at w x h and (about) fps, with its first frame; None otherwise."""
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return None
        # Order matters with OpenCV's DirectShow backend: a size or fps change sets the card up again in its
        # default format, while a FOURCC change keeps the current size and fps. So the format comes last.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if fmt:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and fmt and (self.fourcc_name(cap) != fmt or (frame.shape[1], frame.shape[0]) != (w, h)
                           or cap.get(cv2.CAP_PROP_FPS) < fps - 1):
            ok = False  # the card picked something else: try the next format
        if not ok:
            cap.release()
            return None
        return cap, frame

    def __init__(self, index, w, h, fps, tries=4):
        super().__init__()
        # The card can be busy for a moment (another program closing): retry for a few seconds.
        for attempt in range(tries):
            opened = None
            for fmt in self.FORMATS:
                opened = self.open_format(index, w, h, fps, fmt)
                if opened:
                    break
            if opened:
                self.cap, frame = opened
                break
            if attempt < tries - 1:
                print(f"Le boîtier ne répond pas, nouvel essai ({attempt + 2}/{tries})…")
                time.sleep(1.5)
        else:
            who = capture_blockers()
            if who:
                raise RuntimeError("le boîtier de capture est occupé par " + ", ".join(who) + ". Ferme-le puis relance.")
            raise RuntimeError("le boîtier de capture ne donne pas d'image. Vérifie que la console est allumée, "
                               "que le câble USB est bien branché, et qu'aucun autre programme ne l'utilise "
                               "(OBS, logiciel Elgato, appli Caméra, appel vidéo…).")
        self.size = (frame.shape[1], frame.shape[0])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.format = self.fourcc_name(self.cap)
        # Windows converts YUY2/NV12 to RGB with the BT.601 (standard-definition) matrix; HD consoles send BT.709.
        self.needs_709 = self.format in self.UNCOMPRESSED and self.size[1] >= 720

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


class WindowSource(Source):
    """PS Remote Play (or any window) captured with Windows Graphics Capture; see window_source.py."""

    def __init__(self, fragment="PS Remote Play", out_size=(1920, 1080)):
        super().__init__()
        from windows_capture import WindowsCapture
        from window_source import find_windows, user32, video_rect

        wins = find_windows(fragment)
        if not wins:
            raise RuntimeError(f"fenêtre « {fragment} » introuvable. Lance PS Remote Play, connecte-toi à ta PS5, "
                               "attends que l'image du jeu s'affiche, puis relance Capture Smooth.")
        self.hwnd, self.title = wins[0]
        if user32.IsIconic(self.hwnd):
            raise RuntimeError(f"la fenêtre « {self.title} » est réduite. Rouvre-la : elle peut rester derrière "
                               "Capture Smooth, mais pas dans la barre des tâches.")
        self.size, self.fps = out_size, 60.0
        self.crop, self.crop_age = None, 0
        self.first = threading.Event()
        self.control = None
        self.cap = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=self.hwnd)

        @self.cap.event
        def on_frame_arrived(frame, control):
            if not self.running:
                control.stop()
                return
            buf = frame.frame_buffer  # BGRA, only valid during this call
            h, w = buf.shape[:2]
            self.crop_age += 1
            if self.crop is None or self.crop[0] != (w, h) or self.crop_age > 60:  # window moved or resized
                self.crop, self.crop_age = ((w, h), video_rect(self.hwnd, w, h)), 0
            x, y, cw, ch = self.crop[1]
            if cw < 16 or ch < 9:
                return
            interp = cv2.INTER_AREA if cw > self.size[0] else cv2.INTER_LINEAR
            self.publish(cv2.resize(buf[y:y + ch, x:x + cw, :3], self.size, interpolation=interp))
            self.first.set()

        @self.cap.event
        def on_closed():
            self.error = f"La fenêtre « {self.title} » s'est fermée."
            self.running = False

    def start(self):
        self.control = self.cap.start_free_threaded()
        if not self.first.wait(4):
            self.running = False
            raise RuntimeError(f"aucune image ne vient de « {self.title} ». Vérifie que le jeu s'affiche dans "
                               "PS Remote Play et que la fenêtre n'est pas réduite.")

    def join(self, timeout=None):
        if self.control is not None:
            try:
                self.control.stop()
            except Exception:
                pass


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
        cv2.putText(f, f"IMAGE DE TEST {t:6.1f}s", (w - 780, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        cv2.putText(f, "Ce n'est pas la console. Pour jouer : Echap, puis 'Lancer le jeu'.", (40, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (120, 230, 255), 2)
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
        motion = collections.deque(maxlen=8)  # recent thumbnail changes between real frames
        ai_s = real_s = 0.0  # running averages: one in-between, one real frame (seconds)
        k_auto = None
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

            eng.set_options(self.state.sharpness if self.state.sharp_on else 0.0, self.state.hud)
            cut = False
            if prev is not None:
                change = float(np.abs(eng.tiny(cur) - eng.tiny(prev)).mean())
                if self.state.cuts and len(motion) >= 4 and change > CUT_LEVEL \
                        and change > CUT_RATIO * (float(np.median(motion)) + 1.0):
                    cut = True
                    motion.clear()
                    eng.reset_history()
                    with self.stats.lock:
                        self.stats.cuts += 1
                else:
                    motion.append(change)

            k = self.state.mult if self.state.on else 1
            if prev is None or k == 1 or cut:
                t0 = now()
                self.display.push(eng.real(cur), now(), t_in)
                real_s = 0.8 * real_s + 0.2 * (now() - t0)
                prev = cur
                continue

            # Automatic multiplier: if this card cannot make k-1 in-betweens plus the real frame within one
            # game frame, use fewer, evenly spaced, rather than dropping frames (which looks jerky).
            # Going down as soon as it does not fit, back up only with a clear margin, so it does not flicker.
            if k_auto is None or k_auto > k:
                k_auto = k
            if k > 2 and ai_s > 0:
                fit_down = int((period * 0.92 - real_s) / ai_s) + 1
                fit_up = int((period * 0.78 - real_s) / ai_s) + 1
                if fit_down < k_auto:
                    k_auto = max(2, fit_down)
                elif fit_up > k_auto:
                    k_auto = min(k, fit_up)
            else:
                k_auto = k
            self.state.mult_eff = k_auto
            k = k_auto

            # In-betweens first: the real frame is only shown after them, so it can wait.
            eng.set_pair(prev, cur)
            base = None
            next_expected = t_in + period
            for i in range(1, k):
                if i > 1 and now() > next_expected:
                    break  # the next game frame is due: drop the remaining in-betweens to keep latency down
                t0 = now()
                img = eng.interp(i / k)
                if hasattr(img, "event"):  # GPU frame: wait until it is really computed, for honest pacing
                    img.event.synchronize()
                done = now()
                ai_s = (done - t0) if ai_s == 0 else 0.8 * ai_s + 0.2 * (done - t0)
                with self.stats.lock:
                    self.stats.ai_ms += (done - t0) * 1000
                    self.stats.ai_n += 1
                if base is None:
                    base = done
                self.display.push(img, base + (i - 1) * period / k, None)
            t0 = now()
            real = eng.real(cur)
            if hasattr(real, "event"):
                real.event.synchronize()
            real_s = 0.8 * real_s + 0.2 * (now() - t0)
            self.display.push(real, max(now(), base + (k - 1) * period / k), t_in)
            prev = cur


# ---------------------------------------------------------------- display

class Display:
    def __init__(self, size, mode_name, vsync, state, stats, focus_target=None, sharp_scaling=True, fix_709=False):
        """focus_target: window that must keep the keyboard/controller focus (PS Remote Play). Capture Smooth then
        stays on top without ever taking the focus, and reads its keys directly from the keyboard."""
        self.W, self.H = size
        self.state, self.stats = state, stats
        self.focus_target = focus_target
        self.sharp_scaling = sharp_scaling  # F key: Catmull-Rom scaling, or plain bilinear to compare
        self.fix_709 = fix_709              # C key: BT.709 colours for a capture card
        self.keys_down = set()
        self.queue = collections.deque()
        self.cond = threading.Condition()

        if not glfw.init():
            raise RuntimeError("OpenGL indisponible")
        mon = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(mon)
        if focus_target:
            glfw.window_hint(glfw.FOCUS_ON_SHOW, False)
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
        if focus_target:
            self.no_focus()

        glClearColor(0, 0, 0, 1)
        glEnable(GL_TEXTURE_2D)
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)  # scaling reads past the edges
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.W, self.H, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        try:
            from present import Presenter
            self.presenter = Presenter()
        except Exception as e:  # very old driver: hardware bilinear, no colour fix
            print(f"(mise à l'échelle fine indisponible : {e})")
            self.presenter = None

        # Small on-screen counter (top-left), toggled with F1 or I.
        self.snapshot = None  # (path, time): save one displayed frame, for checking the app
        self.audio = None
        self.interop = None
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
        w, h = 500, 98
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

    def no_focus(self):
        """Always on top, never activated (clicks do not take the focus), focus handed back to the target."""
        import ctypes
        u32 = ctypes.windll.user32
        hwnd = glfw.get_win32_window(self.win)
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOPMOST = -20, 0x08000000, 0x00000008
        u32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, u32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_TOPMOST)
        u32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)  # HWND_TOPMOST, no move/size/activate
        u32.SetForegroundWindow(self.focus_target)
        self.own_hwnd = hwnd

    # Virtual-key codes read in Remote Play mode (the keyboard focus stays with Remote Play).
    GLOBAL_KEYS = {0x1B: "quit", 0x70: "counter", 0x49: "counter", 0x4D: "mute", 0x20: "ai", 0x53: "sharp",
                   0x48: "hud", 0x46: "scaling", 0x43: "color", 0x32: 2, 0x33: 3, 0x34: 4, 0x62: 2, 0x63: 3, 0x64: 4}

    def poll_global_keys(self):
        import ctypes
        u32 = ctypes.windll.user32
        fg = u32.GetForegroundWindow()
        if fg not in (self.focus_target, getattr(self, "own_hwnd", None)):
            self.keys_down.clear()
            return  # typing in another program must not drive Capture Smooth
        for vk, action in self.GLOBAL_KEYS.items():
            down = bool(u32.GetAsyncKeyState(vk) & 0x8000)
            if down and vk not in self.keys_down:
                self.act(action)
            (self.keys_down.add if down else self.keys_down.discard)(vk)

    def on_key(self, win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        names = {glfw.KEY_ESCAPE: "quit", glfw.KEY_F1: "counter", glfw.KEY_I: "counter", glfw.KEY_M: "mute",
                 glfw.KEY_SEMICOLON: "mute", glfw.KEY_SPACE: "ai", glfw.KEY_S: "sharp", glfw.KEY_H: "hud",
                 glfw.KEY_F: "scaling", glfw.KEY_C: "color",
                 glfw.KEY_2: 2, glfw.KEY_3: 3, glfw.KEY_4: 4, glfw.KEY_KP_2: 2, glfw.KEY_KP_3: 3, glfw.KEY_KP_4: 4}
        if key in names:  # M sits on ";" with an AZERTY keyboard
            self.act(names[key])

    def act(self, key):
        if key == "quit":
            glfw.set_window_should_close(self.win, True)
        elif key == "counter":
            self.overlay_on = not self.overlay_on
        elif key == "mute":
            if self.audio:
                self.audio.muted = not self.audio.muted
                print(f"\nSon {'coupé' if self.audio.muted else 'activé'}")
        elif key == "ai":
            self.state.on = not self.state.on
            print(f"\nInterpolation {'ACTIVÉE' if self.state.on else 'DÉSACTIVÉE'}")
        elif key == "sharp":
            self.state.sharp_on = not self.state.sharp_on
            if self.state.sharp_on and self.state.sharpness <= 0:
                self.state.sharpness = 0.3
            print(f"\nNetteté {'ACTIVÉE' if self.state.sharp_on else 'DÉSACTIVÉE'}")
        elif key == "hud":
            self.state.hud = not self.state.hud
            print(f"\nProtection du HUD {'ACTIVÉE' if self.state.hud else 'DÉSACTIVÉE'}")
        elif key == "scaling":
            self.sharp_scaling = not self.sharp_scaling
            print(f"\nMise à l'échelle {'FINE (Catmull-Rom)' if self.sharp_scaling else 'BILINÉAIRE (ancienne)'}")
        elif key == "color":
            self.fix_709 = not self.fix_709
            print(f"\nCouleurs {'BT.709 (HD)' if self.fix_709 else 'BT.601 (Windows)'}")
        elif key in (2, 3, 4):
            self.state.mult = key
            print(f"\nMultiplicateur x{self.state.mult}")

    def show_message(self, title, line):
        """A black screen with a short text, while the AI gets ready."""
        img = np.zeros((self.H, self.W, 4), np.uint8)
        img[..., 3] = 255
        s = self.H / 1080
        for text, y, size, color in ((title, 0.46, 2.2, (255, 255, 255, 255)), (line, 0.56, 1.3, (200, 190, 255, 255))):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size * s, max(2, int(3 * s)))
            cv2.putText(img, text, ((self.W - tw) // 2, int(self.H * y)), cv2.FONT_HERSHEY_DUPLEX, size * s, color,
                        max(2, int(3 * s)), cv2.LINE_AA)
        fw, fh = glfw.get_framebuffer_size(self.win)
        glViewport(0, 0, fw, fh)
        glClear(GL_COLOR_BUFFER_BIT)
        sc = min(fw / self.W, fh / self.H)
        vw, vh = int(self.W * sc), int(self.H * sc)
        glViewport((fw - vw) // 2, (fh - vh) // 2, vw, vh)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.W, self.H, GL_RGBA, GL_UNSIGNED_BYTE, img)
        self.draw_quad()
        glfw.swap_buffers(self.win)
        glfw.poll_events()

    def enable_interop(self):
        """Frames from the CUDA engine go straight into the texture (see gl_interop.py)."""
        from OpenGL.GL import GL_RENDERER, glGetString
        if b"NVIDIA" not in (glGetString(GL_RENDERER) or b""):
            raise RuntimeError("l'affichage ne passe pas par la carte NVIDIA")
        from gl_interop import CudaGLTexture
        glBindTexture(GL_TEXTURE_2D, self.tex)
        self.interop = CudaGLTexture(int(self.tex))

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
        if hasattr(img, "event"):  # frame still on the GPU: straight into the texture
            img.event.synchronize()
            self.interop.upload(img.tensor.data_ptr(), self.W, self.H)
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.W, self.H, GL_RGBA, GL_UNSIGNED_BYTE, img)  # RGBA: 5x faster than BGR
        if self.presenter:
            self.presenter.begin(self.W, self.H, vw, vh, self.sharp_scaling, self.fix_709)
        self.draw_quad()
        if self.presenter:
            self.presenter.end()
        self.draw_overlay(ix, iy, vw, vh)
        if self.snapshot and now() >= self.snapshot[1]:
            from OpenGL.GL import GL_RGB, glReadPixels
            raw = glReadPixels(0, 0, fw, fh, GL_RGB, GL_UNSIGNED_BYTE)
            shot = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3)[::-1, :, ::-1]
            cv2.imwrite(self.snapshot[0], shot)
            self.snapshot = None
        glfw.swap_buffers(self.win)

    def loop(self):
        last_poll = 0.0
        while not glfw.window_should_close(self.win) and self.state.running:
            glfw.poll_events()
            if self.focus_target and now() - last_poll > 0.02:
                last_poll = now()
                self.poll_global_keys()
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


_instance_lock = None


def single_instance():
    """False if another Capture Smooth already has the capture card (two windows would fight over it)."""
    global _instance_lock
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _instance_lock = k32.CreateMutexW(None, False, "Local\\CaptureSmooth")
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


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
    class Parser(argparse.ArgumentParser):
        def error(self, msg):  # shown in a message box, not lost in a console the .exe does not have
            raise RuntimeError(f"option invalide : {msg}")

    ap = Parser(description=f"{APP} : boîtier de capture + images intermédiaires par IA")
    ap.add_argument("--res", choices=RES, default="1440p", help="résolution de capture (défaut 1440p)")
    ap.add_argument("--fps", type=int, default=60, help="fps de capture (défaut 60)")
    ap.add_argument("--x", type=int, choices=[2, 3, 4], default=2, help="multiplicateur d'images (défaut x2)")
    ap.add_argument("--source", choices=["capture", "remoteplay"], default="capture",
                    help="capture = boîtier de capture ; remoteplay = fenêtre PS Remote Play, sans boîtier (plus de latence)")
    ap.add_argument("--window", default="PS Remote Play", help="titre (ou partie du titre) de la fenêtre à capturer")
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
    ap.add_argument("--no-interop", action="store_true", help=argparse.SUPPRESS)  # frames through PC memory (debug)
    ap.add_argument("--nettete", type=int, default=30, help="netteté de 0 à 100 (défaut 30, touche S)")
    ap.add_argument("--no-hud", action="store_true", help="sans protection du HUD (touche H)")
    ap.add_argument("--no-cut", action="store_true", help="interpole aussi sur les changements de plan")
    ap.add_argument("--bilineaire", action="store_true",
                    help="ancienne mise à l'échelle vers l'écran, plus floue (touche F pour comparer)")
    ap.add_argument("--couleurs", choices=["auto", "709", "601"], default="auto",
                    help="matrice de couleurs du boîtier : auto = BT.709 si format non compressé en HD (touche C)")
    if len(sys.argv) == 1:  # double-click on the .exe: settings window
        from launcher import choose
        chosen = choose(APP)
        if chosen is None:
            return
        args = ap.parse_args(chosen)
    else:
        args = ap.parse_args()

    # Only one Capture Smooth may use the capture card (a window source does not need it).
    if args.source == "capture" and not (args.show_taskbar or args.list or args.test) and not single_instance():
        raise RuntimeError("Capture Smooth est déjà ouvert : ferme l'autre fenêtre (Échap) avant d'en lancer une nouvelle.")

    if args.show_taskbar:
        from taskbar import show_taskbar
        show_taskbar()
        message("Barre des tâches réaffichée.")
        return
    if args.list:
        from devices import video_devices
        for i, n in enumerate(video_devices()):
            print(f"{i}: {n}")
        return

    try:  # real pixel coordinates for window capture, whatever the Windows display scaling
        import ctypes
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass

    w, h = RES[args.res]
    focus_target = None
    if args.test:
        src = TestSource(w, h, args.fps, args.test_game_fps)
        print(f"Source : image de test {w}x{h} à {args.fps} Hz (jeu simulé à {args.test_game_fps} fps)")
    elif args.source == "remoteplay":
        src = WindowSource(args.window)
        name = src.title
        focus_target = src.hwnd
        print(f"Source : fenêtre « {name} » (sans boîtier de capture), image ramenée en 1920x1080")
        print("  Attention : Remote Play ajoute son propre retard (compression, réseau, décompression),")
        print("  en général 50 à 100 ms de plus qu'avec un boîtier, et l'image est compressée.")
        print("  Garde la fenêtre Remote Play ouverte (pas réduite) : Capture Smooth s'affiche par-dessus.")
    else:
        idx, name = pick_device(args.device)
        src = CaptureSource(idx, w, h, args.fps)
        print(f"Source : {name} en {src.size[0]}x{src.size[1]} à {src.fps:.0f} fps, format {src.format}")
        if src.format not in CaptureSource.UNCOMPRESSED:
            print("  (format compressé : le boîtier ne passe pas cette résolution et ces fps sans compression)")
        if src.size != (w, h):
            print(f"  (le boîtier a refusé {w}x{h} : vérifie la résolution de sortie de la console)")

    sw, sh = src.size
    work = (sw, sh) if sh <= 1440 else (round(sw * 1440 / sh), 1440)
    scale = 0.5 if (args.quality == "rapide" and work[1] >= 1440) else 1.0

    print(f"{APP} {VERSION} — préparation de l'IA…")
    state = State(args.x, max(0, min(args.nettete, 100)) / 100, not args.no_hud, not args.no_cut)
    stats = Stats()
    fix_709 = args.couleurs == "709" or (args.couleurs == "auto" and getattr(src, "needs_709", False))
    display = Display(work, "fenetre" if args.windowed else args.mode, args.vsync, state, stats, focus_target,
                      sharp_scaling=not args.bilineaire, fix_709=fix_709)
    display.show_message(f"{APP} {VERSION}", "Preparation de l'IA...")
    # The AI warms up in the background (a few seconds) while the window keeps answering Windows.
    built = {}

    def build():
        try:
            built["engine"] = make_engine(args.moteur, src.size, work, scale)
        except Exception as e:
            built["error"] = e

    builder = threading.Thread(target=build, daemon=True)
    builder.start()
    while builder.is_alive():
        glfw.poll_events()
        time.sleep(0.03)
    if "error" in built:
        glfw.terminate()
        from taskbar import show_taskbar
        show_taskbar()
        raise built["error"]
    engine = built["engine"]
    if engine.name == "CUDA" and not args.no_interop:
        try:
            display.enable_interop()
            engine.use_gpu_frames()
        except Exception as e:  # other GPU drives the screen, old driver...: frames go through PC memory
            print(f"(affichage direct GPU indisponible, mode normal : {e})")
    worker = Worker(src, display, state, stats, engine)
    print(f"Prêt. Moteur {engine.name}, image {work[0]}x{work[1]}, IA échelle {scale}, x{args.x}, "
          f"écran {display.refresh} Hz.")
    audio = None
    if not args.test and not args.no_audio and args.source == "capture":  # Remote Play plays its own sound
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
    print("Échap = quitter · Espace = IA on/off · 2/3/4 = multiplicateur · S = netteté · H = HUD · F1 = compteur · M = son\n"
          "F = mise à l'échelle fine/bilinéaire · C = couleurs BT.709/BT.601\n")

    def report():
        while state.running:
            time.sleep(1)
            inp, dups, out, dropped, ai, lat, cuts = stats.snapshot()
            line = (f"\rentrée {inp:3d} img/s (doublons {dups:3d}) | sortie {out:3d} img/s | "
                    f"IA {ai:4.1f} ms | retard ajouté ~{lat:4.1f} ms | sautées {dropped:3d}" +
                    (f" | changement de plan" if cuts else "") + "   ")
            if NO_CONSOLE:  # log file: one line every 10 s is enough
                if int(now()) % 10 == 0:
                    print(line.strip("\r "))
            else:
                print(line, end="", flush=True)
            # Text for the on-screen counter (plain ASCII: the OpenCV font has no accents).
            mode = (f"x{state.mult}" if state.mult_eff >= state.mult else f"x{state.mult_eff} auto") if state.on \
                else "interpolation OFF"
            moving = f"jeu {inp} fps" if inp else "image fixe"
            display.set_info((
                out,
                f"{moving}  {mode}  IA {ai:.0f} ms  retard +{lat:.0f} ms",
                f"{work[0]}x{work[1]}  net {int(state.sharpness * 100) if state.sharp_on else 0}%  "
                f"HUD {'ON' if state.hud else 'OFF'}  son "
                f"{'OFF' if not display.audio or display.audio.muted else 'ON'}  (F1 = masquer)",
            ))
            if src.error:
                print("\n" + src.error)
                state.running = False

    try:
        src.start()
    except Exception:  # the display is already up (taskbar hidden): put things back before reporting
        glfw.terminate()
        from taskbar import show_taskbar
        show_taskbar()
        raise
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


LOG = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "CaptureSmooth" / "capture-smooth.log"
NO_CONSOLE = sys.stdout is None  # the .exe is built without a console window


def setup_log():
    """Without a console, everything printed goes to a log file (kept short: last run only)."""
    if not NO_CONSOLE:
        return
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(LOG, "w", encoding="utf-8", buffering=1)
    except Exception:
        sys.stdout = sys.stderr = open(os.devnull, "w")


def message(text, error=False):
    """A Windows message box (the .exe has no console to show text in)."""
    print(text)
    if NO_CONSOLE:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, APP, 0x10 if error else 0x40)


if __name__ == "__main__":
    setup_log()
    try:
        main()
    except RuntimeError as e:
        msg = str(e)
        message("Erreur : " + msg[:1].upper() + msg[1:], error=True)
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        message(f"Erreur inattendue. Les détails sont dans :\n{LOG}", error=True)
        sys.exit(1)