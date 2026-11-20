"""Settings window shown when the app is started without options (double-click on the .exe).

Returns the command-line options to run with, or None if the window was closed. The last choice is
remembered in %APPDATA%\\CaptureSmooth\\settings.json.
"""
import json
import os
import sys
from pathlib import Path

SETTINGS = Path(os.environ.get("APPDATA", Path.home())) / "CaptureSmooth" / "settings.json"
DEFAULTS = {"source": "capture", "res": "1440p", "game": "30", "mult": "max", "audio": True, "counter": True,
            "sharp": 30, "hud": True}


def load():
    try:
        return {**DEFAULTS, **json.loads(SETTINGS.read_text(encoding="utf-8"))}
    except Exception:
        return dict(DEFAULTS)


def save(s):
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(s), encoding="utf-8")
    except Exception:
        pass


def to_args(s, test=False):
    game = int(s["game"])
    x = (3 if game == 30 else 2) if s["mult"] == "max" else 2
    args = ["--res", s["res"], "--x", str(x)]
    if s.get("source") == "remoteplay" and not test:
        args += ["--source", "remoteplay"]
    if not s["audio"]:
        args.append("--no-audio")
    if not s["counter"]:
        args.append("--hide-counter")
    args += ["--nettete", str(int(s["sharp"]))]
    if not s["hud"]:
        args.append("--no-hud")
    if test:
        args += ["--test", "--windowed", "--test-game-fps", str(game)]
    return args


def choose(title="Capture Smooth"):
    import tkinter as tk
    from tkinter import ttk

    s = load()
    result = {"args": None}
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    try:
        icon = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "assets" / "icon.ico"
        root.iconbitmap(str(icon))
    except Exception:
        pass
    pad = {"padx": 14, "pady": 4}
    frm = ttk.Frame(root, padding=12)
    frm.grid()

    ttk.Label(frm, text=title, font=("Segoe UI", 15, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", **pad)
    ttk.Label(frm, text="Images intermédiaires par IA pour ta console",
              foreground="#666").grid(row=1, column=0, columnspan=2, sticky="w", padx=14)

    source = tk.StringVar(value=s.get("source", "capture"))
    res = tk.StringVar(value=s["res"])
    box = ttk.LabelFrame(frm, text=" Source ", padding=8)
    box.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)
    ttk.Radiobutton(box, text="Boîtier de capture (conseillé)", variable=source, value="capture").grid(
        row=0, column=0, columnspan=3, sticky="w")
    ttk.Label(box, text="Sortie vidéo de la console :").grid(row=1, column=0, sticky="w", padx=(22, 8))
    r1 = ttk.Radiobutton(box, text="1440p ou 4K", variable=res, value="1440p")
    r2 = ttk.Radiobutton(box, text="1080p", variable=res, value="1080p")
    r1.grid(row=1, column=1, sticky="w")
    r2.grid(row=1, column=2, sticky="w", padx=14)
    ttk.Radiobutton(box, text="PS Remote Play (sans boîtier)", variable=source, value="remoteplay").grid(
        row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
    warn = ttk.Label(box, foreground="#b45309", wraplength=440, justify="left",
                     text="Plus de latence : Remote Play ajoute 50 à 100 ms (compression, réseau) en plus de l'IA, "
                          "et l'image est compressée. Lance PS Remote Play et affiche ton jeu avant de cliquer sur "
                          "Lancer ; garde sa fenêtre ouverte (pas réduite).")
    warn.grid(row=3, column=0, columnspan=3, sticky="w", padx=(22, 0), pady=(2, 0))

    def upd_source(*_):
        rp = source.get() == "remoteplay"
        for r in (r1, r2):
            r.state(["disabled"] if rp else ["!disabled"])
        warn.grid() if rp else warn.grid_remove()

    source.trace_add("write", upd_source)

    game = tk.StringVar(value=s["game"])
    box = ttk.LabelFrame(frm, text=" Le jeu tourne à ", padding=8)
    box.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)
    ttk.Radiobutton(box, text="30 fps", variable=game, value="30").grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(box, text="60 fps", variable=game, value="60").grid(row=0, column=1, sticky="w", padx=20)

    mult = tk.StringVar(value=s["mult"])
    box = ttk.LabelFrame(frm, text=" Priorité ", padding=8)
    box.grid(row=4, column=0, columnspan=2, sticky="ew", **pad)
    hint = ttk.Label(box, foreground="#666")
    ttk.Radiobutton(box, text="Fluidité maximale", variable=mult, value="max").grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(box, text="Moins de latence", variable=mult, value="low").grid(row=0, column=1, sticky="w", padx=20)
    hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def update_hint(*_):
        g = int(game.get())
        x = (3 if g == 30 else 2) if mult.get() == "max" else 2
        lat = {(30, 3): 32, (30, 2): 26, (60, 2): 19}.get((g, x), 25)
        hint.config(text=f"Affichage : {g * x} fps  ·  retard ajouté ≈ {lat} ms (RTX 4070 SUPER)")

    for v in (game, mult):
        v.trace_add("write", update_hint)
    update_hint()

    box = ttk.LabelFrame(frm, text=" Image ", padding=8)
    box.grid(row=5, column=0, columnspan=2, sticky="ew", **pad)
    sharp = tk.IntVar(value=int(s["sharp"]))
    sharp_lbl = ttk.Label(box, width=14)
    ttk.Label(box, text="Netteté").grid(row=0, column=0, sticky="w")
    ttk.Scale(box, from_=0, to=100, variable=sharp, length=190,
              command=lambda v: (sharp.set(round(float(v) / 5) * 5), upd_sharp())).grid(row=0, column=1, padx=10)
    sharp_lbl.grid(row=0, column=2, sticky="w")

    def upd_sharp(*_):
        v = sharp.get()
        sharp_lbl.config(text="désactivée" if v == 0 else f"{v} %" + ("  (conseillé)" if v == 30 else ""))

    upd_sharp()
    upd_source()
    hud = tk.BooleanVar(value=s["hud"])
    ttk.Checkbutton(box, text="Protéger le HUD (mini-carte, textes fixes)", variable=hud).grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

    audio = tk.BooleanVar(value=s["audio"])
    counter = tk.BooleanVar(value=s["counter"])
    ttk.Checkbutton(frm, text="Son du boîtier", variable=audio).grid(row=6, column=0, sticky="w", **pad)
    ttk.Checkbutton(frm, text="Compteur fps (F1)", variable=counter).grid(row=6, column=1, sticky="w", **pad)

    ttk.Label(frm, text="Ferme OBS avant de lancer.  Échap = quitter · Espace = IA on/off · S = netteté · M = son",
              foreground="#666").grid(row=7, column=0, columnspan=2, sticky="w", padx=14, pady=(8, 4))

    def current():
        return {"source": source.get(), "res": res.get(), "game": game.get(), "mult": mult.get(), "audio": audio.get(),
                "counter": counter.get(), "sharp": sharp.get(), "hud": hud.get()}

    def go(test=False):
        c = current()
        save(c)
        result["args"] = to_args(c, test)
        root.destroy()

    btns = ttk.Frame(frm)
    btns.grid(row=8, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 2))
    # One obvious button to play; the test picture is only a small link, so it is not clicked by mistake.
    start = tk.Button(btns, text="▶   Lancer le jeu", command=go, font=("Segoe UI", 12, "bold"), fg="white",
                      bg="#6d4ce0", activebackground="#8466f0", activeforeground="white", relief="flat",
                      cursor="hand2", padx=26, pady=8, borderwidth=0)
    start.pack(side="right")
    test = tk.Label(btns, text="Pas de console ? Voir l'image de test", fg="#5b6080", cursor="hand2",
                    font=("Segoe UI", 9, "underline"))
    test.pack(side="left", pady=(14, 0))
    test.bind("<Button-1>", lambda e: go(True))
    root.bind("<Return>", lambda e: go())
    root.bind("<Escape>", lambda e: root.destroy())
    start.focus_set()

    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")
    root.mainloop()
    return result["args"]


if __name__ == "__main__":
    print(choose())
