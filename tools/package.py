"""Zips a build from dist\\ into release files of at most 1.9 GB each (GitHub's limit is 2 GiB per file).

All parts hold paths under the same top folder, so extracting every part in the same place rebuilds it.

    python tools/package.py CaptureSmooth-NVIDIA
"""
import sys
import zipfile
import zlib
from pathlib import Path

LIMIT = 1_900_000_000
ROOT = Path(__file__).resolve().parents[1]


def compressed_size(path, level=6):
    """Exact deflate size of a file, without writing anything."""
    c = zlib.compressobj(level, zlib.DEFLATED, -15)
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            n += len(c.compress(chunk))
    return n + len(c.flush())


def main(name):
    src = ROOT / "dist" / name
    out = ROOT / "dist" / "release"
    out.mkdir(exist_ok=True)
    files = sorted((p for p in src.rglob("*") if p.is_file()), key=lambda p: -p.stat().st_size)
    # Greedy packing: biggest files first, each into the first part where it fits.
    parts = []  # [total, [files]]
    for p in files:
        size = compressed_size(p) if p.stat().st_size > 8_000_000 else p.stat().st_size
        for part in parts:
            if part[0] + size + 1000 < LIMIT:
                part[0] += size + 1000
                part[1].append(p)
                break
        else:
            parts.append([size + 1000, [p]])
    names = []
    for i, (_, members) in enumerate(parts, 1):
        zname = out / (f"{name}-part{i}of{len(parts)}.zip" if len(parts) > 1 else f"{name}.zip")
        with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(members):
                z.write(p, Path(name) / p.relative_to(src))
        names.append(zname)
        print(f"{zname.name}: {zname.stat().st_size / 1e9:.2f} Go, {len(members)} fichiers")
    return names


if __name__ == "__main__":
    main(sys.argv[1])
