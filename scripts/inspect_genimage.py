"""
Inspect a downloaded GenImage subset: counts, corrupt files, file format,
image size and JPEG chroma subsampling for every <generator>/{train,val}/{ai,nature}.

Usage:
    python inspect_genimage.py --root /data/GenImage --gens sdv14 biggan adm glide --sample 500

--gens are the folder names directly under --root (whatever the unzipped
folder is called). The script looks up to 3 levels down for the folder
that contains train/ and/or val/ (test-only generators have no train/).
"""
import argparse
import collections
import os
import random
from pathlib import Path

from PIL import Image, JpegImagePlugin

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0", -1: "n/a"}


def find_split_root(gen_dir: Path, max_depth: int = 3) -> Path:
    frontier = [gen_dir]
    for _ in range(max_depth + 1):
        nxt = []
        for d in frontier:
            if (d / "train").is_dir() or (d / "val").is_dir():
                return d
            try:
                nxt += [Path(e.path) for e in os.scandir(d) if e.is_dir()]
            except PermissionError:
                pass
        frontier = nxt
    raise FileNotFoundError(f"No folder with train/ or val/ found under {gen_dir}")


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return [e.path for e in os.scandir(folder)
            if e.is_file() and os.path.splitext(e.name)[1].lower() in IMG_EXTS]


def inspect_folder(folder: Path, sample: int, rng: random.Random):
    files = list_images(folder)
    picked = files if sample <= 0 or sample >= len(files) else rng.sample(files, sample)
    fmt, sub, sizes, bad = collections.Counter(), collections.Counter(), collections.Counter(), []
    for p in picked:
        try:
            with Image.open(p) as im:
                im.load()
                fmt[im.format] += 1
                s = JpegImagePlugin.get_sampling(im) if im.format == "JPEG" else -1
                sub[SUBSAMPLING.get(s, str(s))] += 1
                sizes[f"{im.size[0]}x{im.size[1]}"] += 1
        except Exception as ex:  # corrupt / truncated
            bad.append((p, repr(ex)))
    return len(files), len(picked), fmt, sub, sizes, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=500, help="images checked per folder; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_bad = []
    for g in args.gens:
        base = find_split_root(Path(args.root) / g)
        print(f"\n=== {g}  ({base}) ===")
        for split in ("train", "val"):
            for cls in ("ai", "nature"):
                n, k, fmt, sub, sizes, bad = inspect_folder(base / split / cls, args.sample, rng)
                all_bad += bad
                print(f"[{split}/{cls}] files={n}  checked={k}  corrupt={len(bad)}")
                print(f"   format     : {dict(fmt)}")
                print(f"   subsampling: {dict(sub)}")
                print(f"   top sizes  : {sizes.most_common(5)}")
    if all_bad:
        with open("corrupt_files.txt", "w") as f:
            for p, e in all_bad:
                f.write(f"{p}\t{e}\n")
        print(f"\n{len(all_bad)} corrupt files written to corrupt_files.txt")


if __name__ == "__main__":
    main()
