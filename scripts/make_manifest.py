"""
Build fixed split manifests (CSV) for the project, so every teammate trains and
tests on exactly the same images.

Protocol (from the proposal):
  - seen generators  -> train / val drawn from the official train split
                        test_seen drawn from the official val split
  - unseen generators -> test only, drawn from the official val split
  - label: ai = 1 (positive), nature = 0
  - every selected image is opened once (corrupt files skipped)
  - MD5 de-duplication across ALL selected images, so no image appears twice
    and no train image leaks into any test set

Usage:
    python make_manifest.py --root /data/GenImage \
        --seen sdv14 biggan --unseen adm glide \
        --n-train 10000 --n-val 1000 --n-test -1 --out manifests

--n-* are per class (ai / nature) per generator. --n-test -1 = use the whole val split.
Paths in the CSV are relative to --root, so teammates only need the same root layout.
"""
import argparse
import csv
import hashlib
import os
import random
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_split_root(gen_dir: Path, max_depth: int = 3) -> Path:
    frontier = [gen_dir]
    for _ in range(max_depth + 1):
        nxt = []
        for d in frontier:
            if (d / "train").is_dir() or (d / "val").is_dir():
                return d
            nxt += [Path(e.path) for e in os.scandir(d) if e.is_dir()]
        frontier = nxt
    raise FileNotFoundError(f"No folder with train/ or val/ found under {gen_dir}")


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(e.path for e in os.scandir(folder)
                  if e.is_file() and os.path.splitext(e.name)[1].lower() in IMG_EXTS)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def ok_image(path):
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def draw(files, n, rng, seen_hashes, stats):
    """Shuffle and take n valid, never-seen-before images (n < 0 -> all)."""
    files = files[:]
    rng.shuffle(files)
    out = []
    for p in files:
        if 0 <= n <= len(out):
            break
        if not ok_image(p):
            stats["corrupt"] += 1
            continue
        h = md5(p)
        if h in seen_hashes:
            stats["duplicate"] += 1
            continue
        seen_hashes.add(h)
        out.append(p)
    if n >= 0 and len(out) < n:
        print(f"  WARNING: wanted {n}, only got {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seen", nargs="+", required=True)
    ap.add_argument("--unseen", nargs="*", default=[])
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="manifests")
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    seen_hashes = set()
    stats = {"corrupt": 0, "duplicate": 0}
    rows = {"train": [], "val": [], "test_seen": []}
    unseen_rows = {g: [] for g in args.unseen}

    def rel(p):
        return os.path.relpath(p, root)

    # Test sets first, so de-dup always removes from train/val, never from test.
    for g in args.seen + args.unseen:
        base = find_split_root(root / g)
        for cls, label in (("ai", 1), ("nature", 0)):
            print(f"[test] {g}/{cls}")
            picked = draw(list_images(base / "val" / cls), args.n_test, rng, seen_hashes, stats)
            target = rows["test_seen"] if g in args.seen else unseen_rows[g]
            target += [(rel(p), label, g) for p in picked]

    for g in args.seen:
        base = find_split_root(root / g)
        for cls, label in (("ai", 1), ("nature", 0)):
            print(f"[train/val] {g}/{cls}")
            picked = draw(list_images(base / "train" / cls),
                          args.n_train + args.n_val, rng, seen_hashes, stats)
            rows["val"] += [(rel(p), label, g) for p in picked[:args.n_val]]
            rows["train"] += [(rel(p), label, g) for p in picked[args.n_val:]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_sets = {**rows, **{f"test_unseen_{g}": r for g, r in unseen_rows.items()}}
    for name, r in all_sets.items():
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "generator"])
            w.writerows(r)
        n_ai = sum(1 for x in r if x[1] == 1)
        print(f"{name:22s} total={len(r):6d}  ai={n_ai:6d}  nature={len(r) - n_ai:6d}")
    with open(out / "README.txt", "w") as f:
        f.write(f"root={root}\nseen={args.seen}\nunseen={args.unseen}\nseed={args.seed}\n"
                f"n_train={args.n_train} n_val={args.n_val} n_test={args.n_test}\n"
                f"skipped corrupt={stats['corrupt']} duplicate={stats['duplicate']}\n")
    print(f"skipped: {stats}")


if __name__ == "__main__":
    main()
