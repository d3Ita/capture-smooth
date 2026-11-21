"""Optional download of NVIDIA Video Frame Generation (package nvidia-vfx, NVIDIA proprietary licence).

Capture Smooth is MIT-licensed and NVIDIA's licence does not clearly allow shipping the SDK inside it, so it is not
in the installer: on an RTX card the user can download it from NVIDIA's own package index, after accepting NVIDIA's
licence in the settings window. One pinned version, checked against its published SHA-256; only the package folder
and its licence files are kept, in %LOCALAPPDATA%\\CaptureSmooth\\nvidia-vfx (engines.vfg_available adds it to the
import path).
"""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from engines import VFG_DIR

VERSION = "0.2.0.0"
URL = f"https://pypi.nvidia.com/nvidia-vfx/nvidia_vfx-{VERSION}-cp312-abi3-win_amd64.whl"
SHA256 = "cebaa12411f7e22b39293205c5c4d07b8073d14ca1e0e7727ee4375207a36445"  # from pypi.nvidia.com/nvidia-vfx/
SIZE = 435_807_359
LICENCES = [  # what the package ships as its licence (also kept next to it once downloaded)
    ("Contrat de licence logicielle NVIDIA",
     "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/"),
    ("Conditions propres aux produits IA NVIDIA",
     "https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/"),
]


class Cancelled(Exception):
    pass


def installed():
    return (VFG_DIR / "nvvfx" / "__init__.py").is_file()


def download(progress=None, cancel=None, url=URL, sha256=SHA256):
    """Downloads, checks and unpacks the package. progress(done_bytes, total_bytes); cancel: threading.Event."""
    VFG_DIR.parent.mkdir(parents=True, exist_ok=True)
    part = VFG_DIR.parent / "nvidia-vfx.whl.part"
    digest = hashlib.sha256()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CaptureSmooth"})
        with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or SIZE)
            done = 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if digest.hexdigest() != sha256:
            raise RuntimeError("le fichier téléchargé ne correspond pas à celui publié par NVIDIA (empreinte SHA-256)")
        unpack(part)
    finally:
        part.unlink(missing_ok=True)


def unpack(wheel):
    """Keeps nvvfx/ and the licence files; unpacks next to the target, then swaps, so a failure leaves nothing half done."""
    tmp = VFG_DIR.with_name(VFG_DIR.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            keep = name.startswith("nvvfx/") or "/licenses/" in name
            target = (tmp / name).resolve()
            if keep and not name.endswith("/") and tmp.resolve() in target.parents:  # no path tricks
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    if not (tmp / "nvvfx" / "__init__.py").is_file():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("paquet NVIDIA inattendu (dossier nvvfx absent)")
    remove()
    tmp.rename(VFG_DIR)


def remove():
    shutil.rmtree(VFG_DIR, ignore_errors=True)


def licence_folder():
    found = list(Path(VFG_DIR).glob("*.dist-info/licenses"))
    return found[0] if found else None
