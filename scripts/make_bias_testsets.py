"""
Build bias-controlled and degraded test sets from an existing test manifest,
so every method can be evaluated where the raw test set is saturated.

Each variant writes its images under <root>/bias_tests/<variant>/{ai,nature}/ and a
manifest <manifests>/test_bias_<variant>.csv with paths relative to <root>.
train_baseline.py evaluates every test_*.csv it finds, so no other change is needed.
Images a variant does not touch are referenced in place (not copied).

Variants
  format / chroma (does the detector rely on PNG-vs-JPEG and chroma subsampling?)
    fake_jpeg444      fakes -> JPEG Q96, 4:4:4      (look like most ImageNet reals); reals untouched
    fake_jpeg420      fakes -> JPEG Q96, 4:2:0                                     ; reals untouched
    real_jpeg420      reals -> JPEG Q96, 4:2:0                                     ; fakes untouched
    all_jpeg444       both  -> JPEG Q96, 4:4:4      (same format for both classes)
  size (does it rely on the fakes' fixed resolution vs ~500x375 reals?)
    size_aligned      reals -> resized (short side) + center-cropped to THIS generator's fake size
                      (BigGAN 128, ADM 256, SD 512 ...), saved PNG like the fakes
    size_jpeg_aligned same size alignment, then both classes -> JPEG Q96 4:4:4 (format + size aligned)
    real_down128      reals -> 128x128 for every generator (for ADM this makes reals SMALLER
                      than fakes: a "reversed size" probe; equals size_aligned for BigGAN)
  degradations (proposal robustness tests), applied to both classes
    jpeg90 / jpeg70 / jpeg50        JPEG with that quality, 4:2:0
    blur0.5 / blur1 / blur2         Gaussian blur sigma, saved PNG
    down75 / down50 / down25        downscale to that % of size, saved PNG

Usage:
    python scripts\\make_bias_testsets.py --root data\\GenImage_small --manifests manifests_smoke
    python scripts\\make_bias_testsets.py --root data\\GenImage_small --manifests manifests_smoke --variants fake_jpeg444 all_jpeg444
    --source test_seen (default) or e.g. test_unseen_adm ; --max-per-class 2000 (default), -1 = all
"""
import argparse
import csv
import os
import random
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageFilter

BIAS = ["fake_jpeg444", "fake_jpeg420", "real_jpeg420", "all_jpeg444",
        "size_aligned", "size_jpeg_aligned", "real_down128"]
DEGRADE = ["jpeg90", "jpeg70", "jpeg50", "blur0.5", "blur1", "blur2",
           "down75", "down50", "down25"]
ALL = BIAS + DEGRADE


def center_crop_square(im, size):
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    return im.crop((left, top, left + size, top + size))


def plan(variant, label, fake_size):
    """Return a list of ops for this image, or None if the image stays untouched."""
    fake = label == 1
    if variant == "fake_jpeg444":
        return [("jpeg", 96, 0)] if fake else None
    if variant == "fake_jpeg420":
        return [("jpeg", 96, 2)] if fake else None
    if variant == "real_jpeg420":
        return None if fake else [("jpeg", 96, 2)]
    if variant == "all_jpeg444":
        return [("jpeg", 96, 0)]
    if variant == "size_aligned":
        return None if fake else [("crop", fake_size), ("png",)]
    if variant == "size_jpeg_aligned":
        return [("jpeg", 96, 0)] if fake else [("crop", fake_size), ("jpeg", 96, 0)]
    if variant == "real_down128":
        return None if fake else [("crop", 128), ("png",)]
    if variant.startswith("jpeg"):
        return [("jpeg", int(variant[4:]), 2)]
    if variant.startswith("blur"):
        return [("blur", float(variant[4:])), ("png",)]
    if variant.startswith("down"):
        return [("down", int(variant[4:]) / 100), ("png",)]
    raise ValueError(variant)


def apply(job):
    src, dst, ops = job
    if os.path.exists(dst):
        return True
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            for op in ops:
                if op[0] == "crop":
                    im = center_crop_square(im, op[1])
                elif op[0] == "blur":
                    im = im.filter(ImageFilter.GaussianBlur(op[1]))
                elif op[0] == "down":
                    w, h = im.size
                    im = im.resize((max(1, round(w * op[1])), max(1, round(h * op[1]))), Image.BICUBIC)
                elif op[0] == "jpeg":
                    im.save(dst, "JPEG", quality=op[1], subsampling=op[2])
                elif op[0] == "png":
                    im.save(dst, "PNG")
        return True
    except Exception as ex:
        print(f"  failed {src}: {ex!r}")
        return False


def read_manifest(path, max_per_class, seed):
    with open(path, newline="") as f:
        rows = [(r["path"], int(r["label"]), r["generator"]) for r in csv.DictReader(f)]
    if max_per_class > 0:
        rng = random.Random(seed)
        out = []
        for lab in (1, 0):
            cls = [r for r in rows if r[1] == lab]
            rng.shuffle(cls)
            out += sorted(cls[:max_per_class])
        rows = out
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--source", default="test_seen", help="manifest name without .csv")
    ap.add_argument("--variants", nargs="+", default=ALL, help=f"any of: {' '.join(ALL)}")
    ap.add_argument("--max-per-class", type=int, default=2000, help="-1 = all images")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    root, mdir = Path(args.root), Path(args.manifests)
    rows = read_manifest(mdir / f"{args.source}.csv", args.max_per_class, args.seed)
    tag = "" if args.source == "test_seen" else f"{args.source.replace('test_', '')}_"
    print(f"source {args.source}: {len(rows)} images, variants: {args.variants}")

    # fake resolution per generator (GenImage fakes are square), used by size_aligned
    fake_size = {}
    for rel, label, gen in rows:
        if label == 1 and gen not in fake_size:
            with Image.open(root / rel) as im:
                fake_size[gen] = min(im.size)
    print(f"fake size per generator: {fake_size}")

    for v in args.variants:
        if v not in ALL:
            raise SystemExit(f"unknown variant {v}; choose from {ALL}")
        name = f"{tag}{v}"
        out_rows, jobs = [], []
        for rel, label, gen in rows:
            ops = plan(v, label, fake_size.get(gen, 128))
            if ops is None:
                out_rows.append((rel, label, gen))
                continue
            ext = ".jpg" if any(o[0] == "jpeg" for o in ops) else ".png"
            cls = "ai" if label == 1 else "nature"
            sub = f"{name}_{fake_size.get(gen, 128)}px" if v in ("size_aligned", "size_jpeg_aligned") else name
            new_rel = Path("bias_tests") / sub / cls / f"{gen}__{Path(rel).stem}{ext}"
            (root / new_rel).parent.mkdir(parents=True, exist_ok=True)
            jobs.append((str(root / rel), str(root / new_rel), ops))
            out_rows.append((new_rel.as_posix(), label, gen))
        with Pool(args.workers) as pool:
            ok = pool.map(apply, jobs, chunksize=64)
        good = {j[1] for j, k in zip(jobs, ok) if k}
        out_rows = [r for r in out_rows
                    if not r[0].startswith("bias_tests/") or str(root / r[0]) in good]
        with open(mdir / f"test_bias_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "generator"])
            w.writerows(out_rows)
        print(f"  {name:28s} images={len(out_rows):6d}  newly processed={len(jobs)}")
    print("done. Now run train_baseline.py with the same --manifests; it evaluates every test_*.csv")


if __name__ == "__main__":
    main()
