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
            "sharp": 30, "hud": True, "engine": "vfg", "clean": False, "upscale": True}


def nvidia_build():
    """The NVIDIA build (PyTorch inside) can run NVIDIA Frame Generation; the DirectML one cannot."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


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


def vfg_installed():
    try:
        from vfg_download import installed
        return installed()
    except Exception:
        return False


def vfg_dialog(root, on_done, engine):
    """Explains NVIDIA's module and licence, asks for acceptance, downloads it with a progress bar."""
    import threading
    import tkinter as tk
    import webbrowser
    from tkinter import ttk

    import vfg_download as vd

    win = tk.Toplevel(root)
    win.title("NVIDIA Frame Generation")
    win.resizable(False, False)
    win.transient(root)
    win.grab_set()
    f = ttk.Frame(win, padding=16)
    f.grid()
    ttk.Label(f, text="NVIDIA Frame Generation", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
    ttk.Label(f, wraplength=470, justify="left", text=(
        "Ce module appartient à NVIDIA et n'est pas fourni avec Capture Smooth. Il est téléchargé directement "
        f"depuis le site de NVIDIA (pypi.nvidia.com, version {vd.VERSION}, {vd.SIZE / 2**20:.0f} Mo), vérifié, puis "
        "installé pour ton compte seulement. Il demande une carte GeForce RTX ; sinon Capture Smooth garde RIFE.\n\n"
        "Son utilisation est soumise à la licence de NVIDIA, que tu dois lire et accepter :")).grid(
        row=1, column=0, sticky="w", pady=(6, 4))
    for i, (label, url) in enumerate(vd.LICENCES):
        link = tk.Label(f, text=label, fg="#3557c4", cursor="hand2", font=("Segoe UI", 9, "underline"))
        link.grid(row=2 + i, column=0, sticky="w", padx=(12, 0))
        link.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
    accept = tk.BooleanVar(value=False)
    ttk.Checkbutton(f, text="J'ai lu la licence de NVIDIA et je l'accepte", variable=accept).grid(
        row=4, column=0, sticky="w", pady=(10, 4))
    bar = ttk.Progressbar(f, length=470, maximum=1000)
    bar.grid(row=5, column=0, sticky="ew", pady=(4, 2))
    status = ttk.Label(f, foreground="#666", text="")
    status.grid(row=6, column=0, sticky="w")
    btns = ttk.Frame(f)
    btns.grid(row=7, column=0, sticky="e", pady=(10, 0))
    go = ttk.Button(btns, text="Télécharger", state="disabled")
    go.pack(side="right")
    close = ttk.Button(btns, text="Annuler")
    close.pack(side="right", padx=8)
    accept.trace_add("write", lambda *_: go.config(state="normal" if accept.get() else "disabled"))

    state = {"done": 0, "total": vd.SIZE, "result": None, "thread": None}
    cancel = threading.Event()

    def work():
        try:
            vd.download(lambda d, t: state.update(done=d, total=t), cancel)
            state["result"] = "ok"
        except vd.Cancelled:
            state["result"] = "cancelled"
        except Exception as e:
            state["result"] = f"Échec : {e}"

    def poll():
        bar["value"] = 1000 * state["done"] / max(state["total"], 1)
        if state["result"] is None:
            if state["done"] >= state["total"]:
                status.config(text="Vérification et installation…")
            else:
                status.config(text=f"{state['done'] / 2**20:.0f} / {state['total'] / 2**20:.0f} Mo")
            win.after(200, poll)
        elif state["result"] == "ok":
            engine.set("vfg")
            win.destroy()
            on_done()
        elif state["result"] == "cancelled":
            win.destroy()
        else:
            status.config(text=state["result"], foreground="#b91c1c")
            go.config(state="normal")

    def start():
        go.config(state="disabled")
        state.update(done=0, result=None)
        state["thread"] = threading.Thread(target=work, daemon=True)
        state["thread"].start()
        poll()

    def stop():
        if state["thread"] is not None and state["result"] is None:
            cancel.set()  # poll() closes the window once the download stopped
            status.config(text="Annulation…")
        else:
            win.destroy()

    go.config(command=start)
    close.config(command=stop)
    win.protocol("WM_DELETE_WINDOW", stop)


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
    if s.get("engine") == "rife":
        args += ["--moteur", "rife"]
    if s.get("clean", False):
        args += ["--nettoyage", "oui"]
    if not s.get("upscale", True):
        args += ["--agrandissement", "shader"]
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
    engine = tk.StringVar(value=s.get("engine", "vfg"))
    clean = tk.BooleanVar(value=s.get("clean", False))
    upscale = tk.BooleanVar(value=s.get("upscale", True))

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
        # Measured with the test picture at 1440p (RTX 4070 SUPER).
        vfg = engine.get() == "vfg" and nvidia_build() and vfg_installed()
        table = {(30, 3): 30, (30, 2): 25, (60, 2): 15} if vfg else {(30, 3): 32, (30, 2): 27, (60, 2): 18}
        hint.config(text=f"Affichage : {g * x} fps  ·  retard ajouté ≈ {table.get((g, x), 25)} ms (RTX 4070 SUPER)")

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

    row = 6
    if nvidia_build():
        ebox = ttk.LabelFrame(frm, text=" Moteur IA ", padding=8)
        ebox.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        row += 1

        def render_engine():
            from vfg_download import installed
            for w in ebox.winfo_children():
                w.destroy()
            if installed():
                ttk.Radiobutton(ebox, text="NVIDIA Frame Generation (cartes RTX) : plus fidèle et plus rapide",
                                variable=engine, value="vfg").grid(row=0, column=0, sticky="w")
                ttk.Radiobutton(ebox, text="RIFE (intégré)", variable=engine, value="rife").grid(
                    row=1, column=0, sticky="w")
                ttk.Label(ebox, text="PS Remote Play (cartes RTX) :", foreground="#444").grid(
                    row=2, column=0, sticky="w", pady=(8, 0))
                ttk.Checkbutton(ebox, text="Agrandissement IA vers ton écran (plus net, touche U)",
                                variable=upscale).grid(row=3, column=0, sticky="w", padx=(12, 0))
                ttk.Checkbutton(ebox, text="Nettoyage de la compression (touche N ; utile seulement si l'image "
                                           "est très dégradée)", variable=clean).grid(
                    row=4, column=0, sticky="w", padx=(12, 0))
            else:
                ttk.Label(ebox, wraplength=440, justify="left", foreground="#444",
                          text="NVIDIA Frame Generation : la génération d'images de NVIDIA, plus fidèle et plus "
                               "rapide que RIFE sur les cartes RTX. Module de NVIDIA à télécharger (416 Mo, licence NVIDIA).").grid(
                    row=0, column=0, sticky="w")
                ttk.Button(ebox, text="Télécharger…", command=lambda: vfg_dialog(root, render_engine, engine)).grid(
                    row=1, column=0, sticky="w", pady=(6, 0))
            update_hint()

        engine.trace_add("write", lambda *_: update_hint())

    audio = tk.BooleanVar(value=s["audio"])
    counter = tk.BooleanVar(value=s["counter"])
    ttk.Checkbutton(frm, text="Son du boîtier", variable=audio).grid(row=row, column=0, sticky="w", **pad)
    ttk.Checkbutton(frm, text="Compteur fps (F1)", variable=counter).grid(row=row, column=1, sticky="w", **pad)

    ttk.Label(frm, text="Ferme OBS avant de lancer.  B = comparer avec l'original · Échap = quitter · M = son",
              foreground="#666").grid(row=row + 1, column=0, columnspan=2, sticky="w", padx=14, pady=(8, 4))

    def current():
        return {"source": source.get(), "res": res.get(), "game": game.get(), "mult": mult.get(), "audio": audio.get(),
                "counter": counter.get(), "sharp": sharp.get(), "hud": hud.get(), "engine": engine.get(),
                "clean": clean.get(), "upscale": upscale.get()}

    def go(test=False):
        c = current()
        save(c)
        result["args"] = to_args(c, test)
        root.destroy()

    if nvidia_build():
        render_engine()
    btns = ttk.Frame(frm)
    btns.grid(row=row + 2, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 2))
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
