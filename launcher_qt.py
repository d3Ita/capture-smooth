"""Settings window (Qt Widgets through QtPy), shown when the app is started without options.

Same contract as launcher.choose(): returns the command-line options to run with, or None if the window was
closed. Settings, defaults and the option list come from launcher.py, so both windows behave the same. Works
offline: nothing here touches the network except the optional NVIDIA download, started by the user.
"""
import ctypes
import os
import sys
import threading
import webbrowser
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")  # the binding shipped with the app (QtPy would try PyQt5 first)
from qtpy.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QAbstractButton, QApplication, QButtonGroup, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from launcher import DEFAULTS, load, nvidia_build, save, to_args, vfg_installed

# GeForce-like look (the style only, no NVIDIA name or logo): near-black panels, lime green accent with black
# text on it, square corners, uppercase headings in Bahnschrift (a DIN-like font that ships with Windows).
ACCENT = "#76b900"
STYLE = """
* { font-family: "Segoe UI"; color: #ececec; }
QWidget#root { background: #0b0b0b; }
QFrame#side { background: #121212; border-right: 1px solid #262626; }
QLabel#appname { font-family: "Bahnschrift"; font-size: 15px; font-weight: 600; letter-spacing: 3px; color: #ffffff; }
QLabel#muted, QLabel#hint { color: #9a9a9a; }
QLabel#hint { font-size: 12px; }
QLabel#h1 { font-family: "Bahnschrift"; font-size: 26px; font-weight: 600; letter-spacing: 1px; color: #ffffff;
    padding-top: 6px; }
QLabel#h2 { font-family: "Bahnschrift"; font-size: 12px; font-weight: 600; color: %(accent)s; letter-spacing: 2px;
    padding-top: 3px; }
QLabel#warn { color: #f5a524; }
QLabel#big { font-family: "Bahnschrift"; font-size: 38px; font-weight: 600; color: #ffffff; }
QPushButton#nav { font-family: "Bahnschrift"; text-align: left; padding: 11px 14px; border: none;
    border-left: 3px solid transparent; font-size: 14px; letter-spacing: 1px; color: #b3b3b3; background: transparent; }
QPushButton#nav:hover { background: #1a1a1a; color: #ffffff; }
QPushButton#nav:checked { background: #1d1d1d; color: #ffffff; border-left: 3px solid %(accent)s; }
QFrame#card { background: #161616; border: 1px solid #262626; border-radius: 2px; }
QPushButton#seg { font-family: "Bahnschrift"; padding: 8px 16px; border: 1px solid #333333; background: #161616;
    color: #c8c8c8; font-size: 13px; letter-spacing: 0.5px; }
QPushButton#seg:hover { background: #202020; color: #ffffff; }
QPushButton#seg:checked { background: %(accent)s; border-color: %(accent)s; color: #000000; font-weight: 600; }
QPushButton#seg:disabled { color: #555555; }
QPushButton#seg:checked:disabled { background: #2a2a2a; border-color: #2a2a2a; color: #808080; }
QPushButton#play { font-family: "Bahnschrift"; background: %(accent)s; color: #000000; font-size: 16px; font-weight: 700;
    letter-spacing: 2px; border: none; border-radius: 2px; padding: 14px 34px; }
QPushButton#play:hover { background: #8bd400; }
QPushButton#play:pressed { background: #5f9600; }
QPushButton#ghost { font-family: "Bahnschrift"; background: transparent; border: 1px solid %(accent)s; color: %(accent)s;
    border-radius: 2px; padding: 8px 18px; letter-spacing: 1px; }
QPushButton#ghost:hover { background: %(accent)s; color: #000000; }
QPushButton#ghost:disabled { color: #555555; border-color: #333333; }
QPushButton#link { background: transparent; border: none; color: #9a9a9a; text-decoration: underline; padding: 4px; }
QPushButton#link:hover { color: %(accent)s; }
QSlider::groove:horizontal { height: 4px; background: #2e2e2e; }
QSlider::sub-page:horizontal { background: %(accent)s; }
QSlider::handle:horizontal { width: 12px; height: 18px; margin: -7px 0; background: #ffffff; border-radius: 1px; }
QSlider::handle:horizontal:hover { background: %(accent)s; }
QCheckBox { spacing: 10px; font-size: 13px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 1px; border: 1px solid #555555; background: #161616; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; }
QProgressBar { background: #2e2e2e; border: none; border-radius: 0; height: 6px; text-align: center; }
QProgressBar::chunk { background: %(accent)s; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 8px; background: transparent; }
QScrollBar::handle:vertical { background: #333333; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
""" % {"accent": ACCENT}


def asset(name):
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "assets" / name


def gpu_name():
    """Name of the graphics card driving the main screen (Windows, no driver library loaded)."""
    class DisplayDevice(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("DeviceName", ctypes.c_wchar * 32), ("DeviceString", ctypes.c_wchar * 128),
                    ("StateFlags", ctypes.c_ulong), ("DeviceID", ctypes.c_wchar * 128), ("DeviceKey", ctypes.c_wchar * 128)]
    try:
        d = DisplayDevice()
        d.cb = ctypes.sizeof(d)
        i = 0
        while ctypes.windll.user32.EnumDisplayDevicesW(None, i, ctypes.byref(d), 0):
            if d.StateFlags & 0x4:  # DISPLAY_DEVICE_PRIMARY_DEVICE
                return d.DeviceString
            i += 1
    except Exception:
        pass
    return "carte graphique inconnue"


def estimate(game, mult, vfg):
    """Shown fps and added delay, measured with the test picture at 1440p on an RTX 4070 SUPER."""
    x = (3 if game == 30 else 2) if mult == "max" else 2
    table = {(30, 3): 30, (30, 2): 25, (60, 2): 15} if vfg else {(30, 3): 32, (30, 2): 27, (60, 2): 18}
    return game * x, table.get((game, x), 25)


# ------------------------------------------------------------------ small widgets

class Toggle(QAbstractButton):
    """An on/off switch."""

    def __init__(self, checked=False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(42, 24)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        on = self.isChecked()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACCENT if on else "#3a3a3a"))
        p.drawRoundedRect(QRectF(0, 0, 42, 24), 12, 12)
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QPointF(30 if on else 12, 12), 9, 9)


class Segmented(QWidget):
    """A row of joined choice buttons; value() is the chosen key."""
    changed = Signal(str)

    def __init__(self, choices, value):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.buttons = {}
        for i, (key, label) in enumerate(choices):
            b = QPushButton(label)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            r = "2px"
            left = f"border-top-left-radius:{r}; border-bottom-left-radius:{r};" if i == 0 else ""
            right = f"border-top-right-radius:{r}; border-bottom-right-radius:{r};" if i == len(choices) - 1 else ""
            b.setStyleSheet(left + right + ("" if i == 0 else "border-left: none;"))
            self.group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
        lay.addStretch(1)
        self.set(value)

    def set(self, key):
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def value(self):
        return next(k for k, b in self.buttons.items() if b.isChecked())


def card():
    f = QFrame()
    f.setObjectName("card")
    return f


def label(text, name=None, wrap=False):
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


def slider_row(grid, row, title, value, default, on_change=None):
    s = QSlider(Qt.Horizontal)
    s.setRange(0, 100)
    s.setSingleStep(5)
    s.setPageStep(10)
    s.setValue(int(value))
    val = label("", "muted")
    val.setMinimumWidth(110)

    def upd(v):
        v = round(v / 5) * 5
        if v != s.value():
            s.setValue(v)
            return
        val.setText("désactivé" if v == 0 else f"{v} %" + ("  · conseillé" if v == default else ""))
        if on_change:
            on_change(v)

    s.valueChanged.connect(upd)
    upd(s.value())
    grid.addWidget(label(title), row, 0)
    grid.addWidget(s, row, 1)
    grid.addWidget(val, row, 2)
    return s


def toggle_row(lay, title, hint, checked):
    row = QHBoxLayout()
    t = Toggle(checked)
    col = QVBoxLayout()
    col.setSpacing(1)
    col.addWidget(label(title))
    if hint:
        col.addWidget(label(hint, "hint", wrap=True))
    row.addLayout(col, 1)
    row.addWidget(t, 0, Qt.AlignVCenter)
    lay.addLayout(row)
    return t


# ------------------------------------------------------------------ the window

class Window(QWidget):
    def __init__(self, title, version):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(str(asset("icon.ico"))))
        self.setMinimumSize(940, 640)
        self.s = load()
        self.result = None
        self.nvidia = nvidia_build()
        self.found = {"device": None, "gpu": None}

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(220)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(16, 20, 16, 16)
        sl.setSpacing(4)
        logo = QLabel()
        pix = QPixmap(str(asset("logo.png")))
        if not pix.isNull():
            logo.setPixmap(pix.scaledToWidth(186, Qt.SmoothTransformation))
        sl.addWidget(logo)
        sl.addSpacing(6)
        sl.addWidget(label("CAPTURE SMOOTH", "appname"))
        sl.addWidget(label(f"version {version}" if version else "", "hint"))
        sl.addSpacing(24)
        self.pages = QStackedWidget()
        self.nav = QButtonGroup(self)
        for i, (text, page) in enumerate((("JOUER", self.page_play()), ("IMAGE", self.page_picture()),
                                         ("MOTEUR IA", self.page_engine()), ("RACCOURCIS", self.page_keys()))):
            b = QPushButton(text)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.nav.addButton(b, i)
            sl.addWidget(b)
            self.pages.addWidget(self.scroll(page))
        self.nav.idClicked.connect(self.pages.setCurrentIndex)
        self.nav.button(0).setChecked(True)
        sl.addStretch(1)
        sl.addWidget(label("Fonctionne hors ligne.\nFerme OBS et le logiciel du\nboîtier avant de lancer.", "hint"))
        root.addWidget(side)
        root.addWidget(self.pages, 1)

        self.refresh_estimate()
        self.detect()

    @staticmethod
    def scroll(page):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page.setObjectName("root")
        area.setWidget(page)
        return area

    @staticmethod
    def page_frame(title, subtitle):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(36, 30, 36, 28)
        lay.setSpacing(14)
        lay.addWidget(label(title.upper(), "h1"))
        lay.addWidget(label(subtitle, "muted", wrap=True))
        lay.addSpacing(6)
        return page, lay

    # -------------------------------------------------------------- Play
    def page_play(self):
        page, lay = self.page_frame("Prêt à jouer", "Choisis ta source et tes fps, puis lance : l'image du jeu "
                                    "s'affiche en plein écran avec les images ajoutées par l'IA.")
        status = QHBoxLayout()
        status.setSpacing(12)
        self.st_device, self.st_gpu, self.st_engine = [], [], []
        for store, title in ((self.st_device, "BOÎTIER"), (self.st_gpu, "CARTE GRAPHIQUE"), (self.st_engine, "MOTEUR IA")):
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(16, 12, 16, 12)
            cl.setSpacing(2)
            cl.addWidget(label(title, "h2"))
            v = label("recherche…", None, wrap=True)
            v.setStyleSheet("font-size: 14px;")
            cl.addWidget(v)
            store.append(v)
            status.addWidget(c, 1)
        lay.addLayout(status)

        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(20, 18, 20, 18)
        g.setHorizontalSpacing(18)
        g.setVerticalSpacing(14)
        self.source = Segmented((("capture", "Boîtier de capture"), ("remoteplay", "PS Remote Play")),
                                self.s.get("source", "capture"))
        self.res = Segmented((("1440p", "1440p ou 4K"), ("1080p", "1080p")), self.s["res"])
        self.game = Segmented((("30", "30 fps"), ("60", "60 fps")), self.s["game"])
        self.mult = Segmented((("max", "Fluidité maximale"), ("low", "Moins de latence")), self.s["mult"])
        rows = (("Source", self.source), ("Sortie de la console", self.res), ("Le jeu tourne à", self.game),
                ("Priorité", self.mult))
        for i, (t, w) in enumerate(rows):
            g.addWidget(label(t, "muted"), i, 0)
            g.addWidget(w, i, 1)
        self.rp_warn = label("PS Remote Play ajoute 50 à 100 ms de latence (compression, réseau) et l'image est "
                             "compressée. Lance PS Remote Play et affiche ton jeu avant ; garde sa fenêtre ouverte, "
                             "en 1920x1080.", "warn", wrap=True)
        g.addWidget(self.rp_warn, len(rows), 1)
        g.setColumnStretch(1, 1)
        lay.addWidget(c)
        for w in (self.source, self.game, self.mult):
            w.changed.connect(lambda _: self.refresh_estimate())

        bottom = QHBoxLayout()
        est = QVBoxLayout()
        est.setSpacing(0)
        self.est_fps = label("", "big")
        self.est_lat = label("", "muted", wrap=True)
        est.addWidget(self.est_fps)
        est.addWidget(self.est_lat)
        bottom.addLayout(est, 1)
        test = QPushButton("Pas de console ? Voir l'image de test")
        test.setObjectName("link")
        test.setCursor(Qt.PointingHandCursor)
        test.clicked.connect(lambda: self.go(True))
        bottom.addWidget(test, 0, Qt.AlignBottom)
        bottom.addSpacing(12)
        play = QPushButton("LANCER LE JEU")
        play.setObjectName("play")
        play.setCursor(Qt.PointingHandCursor)
        play.setDefault(True)
        play.clicked.connect(lambda: self.go(False))
        bottom.addWidget(play, 0, Qt.AlignBottom)
        lay.addStretch(1)
        lay.addLayout(bottom)
        return page

    def refresh_estimate(self):
        rp = self.source.value() == "remoteplay"
        for b in self.res.buttons.values():
            b.setEnabled(not rp)
        self.rp_warn.setVisible(rp)
        vfg = self.nvidia and vfg_installed() and self.s.get("engine", "vfg") == "vfg"
        fps, lat = estimate(int(self.game.value()), self.mult.value(), vfg)
        self.est_fps.setText(f"{fps} fps")
        self.est_lat.setText(f"affichés · retard ajouté ≈ {lat} ms" + ("  (+ Remote Play)" if rp else "")
                             + "\nmesuré sur RTX 4070 SUPER")
        if self.st_engine:
            if not self.nvidia:
                self.st_engine[0].setText("RIFE · DirectML")
            elif vfg:
                self.st_engine[0].setText("NVIDIA Frame Generation")
            else:
                self.st_engine[0].setText("RIFE · CUDA")

    def detect(self):
        """Shows the capture card and graphics card found by detect_hardware() (before Qt started)."""
        self.found.update(FOUND)
        self.poll = QTimer(self)
        self.poll.timeout.connect(self.show_found)
        self.poll.start(0)

    def show_found(self):
        if self.found["gpu"] is None:
            return
        self.poll.stop()
        dev = self.found["device"]
        self.st_device[0].setText(dev or "aucun boîtier détecté")
        self.st_device[0].setStyleSheet(f"font-size: 14px; color: {ACCENT if dev else '#f5a524'};")
        self.st_gpu[0].setText(self.found["gpu"])

    # -------------------------------------------------------------- Picture
    def page_picture(self):
        page, lay = self.page_frame("Image", "Netteté et effets d'image, appliqués à toutes les images affichées. "
                                    "En jeu : E coupe les effets, B montre l'image d'origine pour comparer.")
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(20, 18, 20, 18)
        g.setHorizontalSpacing(18)
        g.setVerticalSpacing(16)
        g.addWidget(label("NETTETÉ ET EFFETS", "h2"), 0, 0, 1, 3)
        self.sharp = slider_row(g, 1, "Netteté", self.s["sharp"], 30)
        self.fx = {}
        for i, (key, title, default) in enumerate((("deband", "Debanding", 40), ("lcon", "Contraste local", 25),
                                                   ("vib", "Vibrance", 25))):
            self.fx[key] = slider_row(g, 2 + i, title, self.s.get(key, default), default)
        g.addWidget(label("Debanding : enlève les bandes des dégradés (ciel, brouillard). Contraste local : relief des "
                          "textures. Vibrance : ravive les couleurs ternes, garde les tons chair.", "hint", wrap=True),
                    5, 0, 1, 3)
        g.setColumnStretch(1, 1)
        lay.addWidget(c)

        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(20, 18, 20, 18)
        cl.setSpacing(14)
        cl.addWidget(label("AFFICHAGE ET SON", "h2"))
        self.hud = toggle_row(cl, "Protéger le HUD", "Mini-carte et textes fixes repris de la vraie image.",
                              self.s["hud"])
        self.audio = toggle_row(cl, "Son du boîtier", "Joué sur les haut-parleurs du PC, calé sur l'image (F2 / F3).",
                                self.s["audio"])
        self.counter = toggle_row(cl, "Compteur fps", "En haut à gauche ; F1 le masque, D affiche le détail.",
                                  self.s["counter"])
        lay.addWidget(c)
        lay.addStretch(1)
        return page

    # -------------------------------------------------------------- Engine
    def page_engine(self):
        page, lay = self.page_frame("Moteur IA", "Le moteur qui crée les images intermédiaires.")
        self.engine_box = QVBoxLayout()
        lay.addLayout(self.engine_box)
        lay.addStretch(1)
        self.engine_choice = self.s.get("engine", "vfg")
        self.render_engine()
        return page

    def render_engine(self):
        while self.engine_box.count():
            w = self.engine_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        if not self.nvidia:
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(20, 18, 20, 18)
            cl.addWidget(label("ÉDITION DIRECTML", "h2"))
            cl.addWidget(label("RIFE sur DirectX 12 : fonctionne avec toutes les cartes graphiques (AMD, Intel, NVIDIA). "
                               "Pour NVIDIA Frame Generation, installe l'édition NVIDIA.", "muted", wrap=True))
            self.engine_box.addWidget(c)
            return
        if vfg_installed():
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(20, 18, 20, 18)
            cl.setSpacing(12)
            cl.addWidget(label("MOTEUR", "h2"))
            self.engine_seg = Segmented((("vfg", "NVIDIA Frame Generation"), ("rife", "RIFE (intégré)")),
                                        self.engine_choice)
            self.engine_seg.changed.connect(self.set_engine)
            cl.addWidget(self.engine_seg)
            cl.addWidget(label("NVIDIA Frame Generation est plus fidèle et environ 3 ms plus rapide par image "
                               "(cartes RTX). RIFE reste disponible pour comparer.", "hint", wrap=True))
            self.engine_box.addWidget(c)
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(20, 18, 20, 18)
            cl.setSpacing(14)
            cl.addWidget(label("PS REMOTE PLAY", "h2"))
            self.upscale = toggle_row(cl, "Agrandissement IA vers ton écran", "Plus net que l'agrandissement "
                                      "habituel (touche U).", self.s.get("upscale", True))
            self.clean = toggle_row(cl, "Nettoyage de la compression", "Seulement si l'image est très dégradée "
                                    "(touche N).", self.s.get("clean", False))
            self.engine_box.addWidget(c)
        else:
            self.engine_box.addWidget(self.vfg_card())

    def set_engine(self, key):
        self.engine_choice = key
        self.s["engine"] = key
        self.refresh_estimate()

    def vfg_card(self):
        import vfg_download as vd
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(20, 18, 20, 18)
        cl.setSpacing(10)
        cl.addWidget(label("NVIDIA FRAME GENERATION · FACULTATIF", "h2"))
        cl.addWidget(label("La génération d'images de NVIDIA, plus fidèle et plus rapide que RIFE sur les cartes RTX. "
                           "Ce module appartient à NVIDIA et n'est pas fourni avec Capture Smooth : il est téléchargé "
                           f"depuis le site de NVIDIA (pypi.nvidia.com, version {vd.VERSION}, "
                           f"{vd.SIZE / 2 ** 20:.0f} Mo), vérifié, puis installé pour ton compte seulement. C'est la "
                           "seule fonction de l'application qui utilise Internet.", "muted", wrap=True))
        links = QHBoxLayout()
        for text, url in vd.LICENCES:
            b = QPushButton(text)
            b.setObjectName("link")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, u=url: webbrowser.open(u))
            links.addWidget(b)
        links.addStretch(1)
        cl.addLayout(links)
        accept = QCheckBox("J'ai lu la licence de NVIDIA et je l'accepte")
        cl.addWidget(accept)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        bar.hide()
        status = label("", "hint")
        cl.addWidget(bar)
        cl.addWidget(status)
        row = QHBoxLayout()
        go = QPushButton("Télécharger")
        go.setObjectName("ghost")
        go.setEnabled(False)
        stop = QPushButton("Annuler")
        stop.setObjectName("ghost")
        stop.hide()
        row.addWidget(go)
        row.addWidget(stop)
        row.addStretch(1)
        cl.addLayout(row)
        accept.toggled.connect(go.setEnabled)

        state = {"done": 0, "total": vd.SIZE, "result": None}
        cancel = threading.Event()

        def work():
            try:
                vd.download(lambda d, t: state.update(done=d, total=t), cancel)
                state["result"] = "ok"
            except vd.Cancelled:
                state["result"] = "cancelled"
            except Exception as e:
                state["result"] = f"Échec : {e}"

        timer = QTimer(c)

        def poll():
            bar.setValue(int(1000 * state["done"] / max(state["total"], 1)))
            if state["result"] is None:
                status.setText("Vérification et installation…" if state["done"] >= state["total"] else
                               f"{state['done'] / 2 ** 20:.0f} / {state['total'] / 2 ** 20:.0f} Mo")
                return
            timer.stop()
            stop.hide()
            if state["result"] == "ok":
                self.engine_choice = "vfg"
                self.s["engine"] = "vfg"
                self.render_engine()
                self.refresh_estimate()
            elif state["result"] == "cancelled":
                bar.hide()
                status.setText("Téléchargement annulé.")
                go.setEnabled(accept.isChecked())
            else:
                status.setText(state["result"])
                status.setStyleSheet("color: #f5a524;")
                go.setEnabled(True)

        def start():
            go.setEnabled(False)
            accept.setEnabled(False)
            stop.show()
            bar.show()
            cancel.clear()
            state.update(done=0, result=None)
            threading.Thread(target=work, daemon=True).start()
            timer.start(200)

        timer.timeout.connect(poll)
        go.clicked.connect(start)
        stop.clicked.connect(lambda: (cancel.set(), status.setText("Annulation…")))
        return c

    # -------------------------------------------------------------- Keys
    def page_keys(self):
        page, lay = self.page_frame("Raccourcis", "Les touches pendant le jeu (aussi en PS Remote Play).")
        c = card()
        g = QGridLayout(c)
        g.setContentsMargins(20, 18, 20, 18)
        g.setHorizontalSpacing(22)
        g.setVerticalSpacing(10)
        keys = (("Échap", "quitter"), ("B", "image d'origine, sans aucune amélioration (comparer)"),
                ("E", "effets d'image on/off"), ("Espace", "images ajoutées on/off"), ("2 / 3 / 4", "multiplicateur"),
                ("S", "netteté on/off"), ("H", "protection du HUD"), ("F1 ou I", "compteur fps"),
                ("D", "détail des mesures"), ("M", "son on/off"), ("F2 / F3", "son plus tôt / plus tard"),
                ("F", "mise à l'échelle fine / simple"), ("C", "couleurs BT.709 / BT.601"),
                ("U", "agrandissement IA (Remote Play)"), ("N", "nettoyage de la compression (Remote Play)"))
        for i, (k, what) in enumerate(keys):
            key = QLabel(k)
            key.setStyleSheet("background:#222222; border:1px solid #3a3a3a; border-bottom:2px solid #76b900;"
                              "border-radius:2px; padding:3px 10px; font-family:'Bahnschrift'; font-weight:600;")
            key.setAlignment(Qt.AlignCenter)
            g.addWidget(key, i, 0)
            g.addWidget(label(what, "muted"), i, 1)
        g.setColumnStretch(1, 1)
        lay.addWidget(c)
        lay.addStretch(1)
        return page

    # -------------------------------------------------------------- start
    def current(self):
        s = dict(self.s)
        s.update(source=self.source.value(), res=self.res.value(), game=self.game.value(), mult=self.mult.value(),
                 sharp=self.sharp.value(), hud=self.hud.isChecked(), audio=self.audio.isChecked(),
                 counter=self.counter.isChecked(), engine=self.engine_choice,
                 **{k: v.value() for k, v in self.fx.items()})
        if hasattr(self, "upscale") and self.upscale is not None:
            try:
                s.update(upscale=self.upscale.isChecked(), clean=self.clean.isChecked())
            except RuntimeError:  # the engine card was rebuilt
                pass
        return s

    def go(self, test):
        s = self.current()
        save(s)
        self.result = to_args(s, test)
        self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.go(False)
        else:
            super().keyPressEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        try:  # dark Windows title bar (Windows 10 2004+ / 11)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(int(self.winId()), 20, ctypes.byref(ctypes.c_int(1)), 4)
        except Exception:
            pass


FOUND = {"device": "", "gpu": "carte graphique inconnue"}


def detect_hardware():
    """Capture card and graphics card names (about 45 ms). Done on the main thread before Qt starts: DirectShow's
    COM objects made from another thread while the Qt window lived made the process crash or fail on exit."""
    try:
        from devices import best_capture, video_devices
        best = best_capture(video_devices())
        FOUND["device"] = best[1] if best else ""
    except Exception:
        FOUND["device"] = ""
    FOUND["gpu"] = gpu_name()


def choose(title="Capture Smooth", version=""):
    detect_hardware()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    w = Window(title, version)
    w.resize(1000, 680)
    w.show()
    app.exec_() if hasattr(app, "exec_") else app.exec()
    result = w.result
    # Qt is done for this run: tear it down now (its COM/OLE state with it), so the game window that follows
    # starts from a clean process.
    w.deleteLater()
    app.processEvents()
    if hasattr(app, "shutdown"):
        app.shutdown()
    return result


if __name__ == "__main__":
    print(choose(version="dev"))
