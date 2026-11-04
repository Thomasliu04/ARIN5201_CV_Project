"""
One-click GenImage subset preparation for a laptop with limited disk space.

For each generator (smallest first, SD V1.4 last) it:
  1. downloads the generator from the Kaggle mirror (or uses archives you put
     into <tmp>/<gen>/ yourself, e.g. official Baidu / Google Drive split zips)
  2. extracts ONLY the images we keep directly from the zip when possible
     (all of val/, plus --n-train random images per class from train/)
  3. deletes the downloaded / temporary files before moving to the next generator

Images are copied byte-for-byte: no resizing, no re-encoding.
Re-running the script skips generators that are already finished (.done marker).

Requirements:
    pip install kaggle pillow
    Kaggle API token: kaggle.com -> Settings -> API -> Create New Token,
        put kaggle.json in  ~/.kaggle/  (Windows: C:\\Users\\<you>\\.kaggle\\)
    7-Zip only if an archive turns out to be a multi-part (.z01, .z02 ...) zip:
        Windows: install 7-Zip;  Mac: brew install sevenzip

Usage (Windows example):
    python prepare_genimage_laptop.py --tmp D:/genimage_tmp --out D:/GenImage_small
Keep the train split of a generator that was test-only before (e.g. ADM), plus a new test-only one:
    python prepare_genimage_laptop.py --tmp D:/genimage_tmp --out D:/GenImage_small --gens adm glide --train-gens biggan adm
Only some generators:
    python prepare_genimage_laptop.py --tmp D:/genimage_tmp --out D:/GenImage_small --gens biggan
Already downloaded archives yourself (put them in <tmp>/<gen>/):
    python prepare_genimage_laptop.py ... --no-download
"""
import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# name -> Kaggle mirror slug (github.com/vtphatt2/GenImage-mirror)
KAGGLE = {
    "biggan": "vtphatt2/GenImage-BigGAN",
    "adm": "vtphatt2/GenImage-ADM",
    "glide": "vtphatt2/GenImage-glide",
    "sdv14": "vtphatt2/genimage-stable-diffusion-v1-4",
}
DEFAULT_ORDER = ["biggan", "adm", "glide", "sdv14"]   # biggest last
TRAIN_GENS = {"sdv14", "biggan"}                      # seen generators; others are test-only

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_PART = re.compile(r"\.z\d{2,3}$", re.I)


# ---------------------------------------------------------------- helpers
def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def log(msg):
    print(msg, flush=True)


def classify(path_str):
    """Return (split, cls) if path looks like .../train|val/ai|nature/<image>, else None."""
    if os.path.splitext(path_str)[1].lower() not in IMG_EXTS:
        return None
    parts = re.split(r"[\\/]+", path_str)
    for i in range(len(parts) - 2):
        if parts[i] in ("train", "val") and parts[i + 1] in ("ai", "nature"):
            return parts[i], parts[i + 1]
    return None


def select(groups, n_train, rng):
    """groups: {(split, cls): [items]} -> same structure with the items we keep."""
    picked = {}
    for key, items in groups.items():
        items = sorted(items, key=str)
        split, _ = key
        if split == "val":
            picked[key] = items
        elif n_train > 0:
            rng.shuffle(items)
            picked[key] = items[:n_train]
    return picked


def find_7z():
    for name in ("7z", "7zz", "7za"):
        p = shutil.which(name)
        if p:
            return p
    win = Path(r"C:\Program Files\7-Zip\7z.exe")
    return str(win) if win.exists() else None


# ---------------------------------------------------------------- extraction
def extract_images_from_zip(zf, dst, n_train, rng, stats):
    groups = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        key = classify(info.filename)
        if key:
            groups.setdefault(key, []).append(info)
    if not groups:
        return False
    for key, infos in select(groups, n_train, rng).items():
        out = dst / key[0] / key[1]
        out.mkdir(parents=True, exist_ok=True)
        for i, info in enumerate(infos, 1):
            target = out / os.path.basename(info.filename)
            if target.exists():
                continue
            try:
                with zf.open(info) as src, open(target, "wb") as f:
                    shutil.copyfileobj(src, f, 1 << 20)
            except Exception as ex:  # CRC error / truncated member
                stats["bad"] += 1
                target.unlink(missing_ok=True)
                log(f"    skip corrupt {info.filename}: {ex!r}")
            if i % 5000 == 0:
                log(f"    {key[0]}/{key[1]}: {i}/{len(infos)}")
        log(f"  {key[0]}/{key[1]}: {len(infos)} images")
    return True


def copy_images_from_folder(src_root, dst, n_train, rng):
    groups = {}
    for root, _, files in os.walk(src_root):
        for f in files:
            p = os.path.join(root, f)
            key = classify(p)
            if key:
                groups.setdefault(key, []).append(p)
    if not groups:
        return False
    for key, paths in select(groups, n_train, rng).items():
        out = dst / key[0] / key[1]
        out.mkdir(parents=True, exist_ok=True)
        for p in paths:
            target = out / os.path.basename(p)
            if not target.exists():
                shutil.copy2(p, target)
        log(f"  {key[0]}/{key[1]}: {len(paths)} images")
    return True


def process_zip(zip_path, work, dst, n_train, rng, stats, keep_tmp):
    """Single (non-split) zip: extract wanted images directly, or unpack nested archives."""
    log(f"  reading {zip_path.name}  (free {free_gb(work):.1f} GB)")
    with zipfile.ZipFile(zip_path) as zf:
        if extract_images_from_zip(zf, dst, n_train, rng, stats):
            done = True
        else:
            nested = [n for n in zf.namelist()
                      if n.lower().endswith(".zip") or SPLIT_PART.search(n)]
            if not nested:
                raise RuntimeError(f"{zip_path} has no train/val images and no inner archives")
            inner = work / (zip_path.stem + "_inner")
            log(f"  {zip_path.name} contains {len(nested)} inner archive files, unpacking them")
            for n in nested:
                zf.extract(n, inner)
            done = False
    if not keep_tmp:
        zip_path.unlink()
    if not done:
        process_archive_dir(inner, work, dst, n_train, rng, stats, keep_tmp)


def process_archive_dir(folder, work, dst, n_train, rng, stats, keep_tmp):
    zips = sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() == ".zip")
    for main in zips:
        parts = sorted(p for p in main.parent.iterdir()
                       if p.stem == main.stem and SPLIT_PART.search(p.name))
        if not parts:
            process_zip(main, work, dst, n_train, rng, stats, keep_tmp)
            continue
        # multi-part zip -> needs 7-Zip, full extraction, then subset copy
        sevenzip = find_7z()
        if not sevenzip:
            sys.exit("Multi-part zip found but 7-Zip is not installed "
                     "(Windows: install 7-Zip; Mac: brew install sevenzip).")
        out = work / (main.stem + "_extracted")
        log(f"  7-Zip extracting {main.name} + {len(parts)} parts (free {free_gb(work):.1f} GB)")
        subprocess.run([sevenzip, "x", str(main), f"-o{out}", "-y", "-bso0", "-bsp1"], check=True)
        if not keep_tmp:
            for p in parts + [main]:
                p.unlink()
        process_archive_dir(out, work, dst, n_train, rng, stats, keep_tmp)  # in case of nesting
        copy_images_from_folder(out, dst, n_train, rng)
        shutil.rmtree(out, ignore_errors=True)


# ---------------------------------------------------------------- download
def kaggle_download(slug, out):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Please run:  pip install kaggle")
    api = KaggleApi()
    api.authenticate()
    log(f"  downloading {slug} -> {out}  (free {free_gb(out):.1f} GB)")
    api.dataset_download_files(slug, path=str(out), unzip=False, quiet=False)


# ---------------------------------------------------------------- main
def count_images(d):
    return {f"{s}/{c}": len(list((d / s / c).glob("*"))) if (d / s / c).is_dir() else 0
            for s in ("train", "val") for c in ("ai", "nature")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True, help="temporary folder (emptied after each generator)")
    ap.add_argument("--out", required=True, help="final compact dataset folder")
    ap.add_argument("--gens", nargs="+", default=DEFAULT_ORDER, choices=list(KAGGLE))
    ap.add_argument("--train-gens", nargs="+", default=sorted(TRAIN_GENS), choices=list(KAGGLE),
                    help="generators whose train/ split is kept (others: val only)")
    ap.add_argument("--n-train", type=int, default=15000,
                    help="train images kept per class for seen generators (margin over 11k needed)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-download", action="store_true", help="use archives already in <tmp>/<gen>/")
    ap.add_argument("--keep-tmp", action="store_true", help="do not delete downloaded files")
    args = ap.parse_args()

    tmp, out = Path(args.tmp), Path(args.out)
    tmp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    for gen in args.gens:
        dst = out / gen
        done_marker = dst / ".done"
        if done_marker.exists() and gen in args.train_gens and "'train/ai': 0" in done_marker.read_text():
            log(f"\n=== {gen}: has val only, now also keeping train ===")
            done_marker.unlink()
        if done_marker.exists():
            log(f"\n=== {gen}: already done, skipping ===")
            continue
        log(f"\n=== {gen} ===  free space: {free_gb(tmp):.1f} GB")
        n_train = args.n_train if gen in args.train_gens else 0
        rng = random.Random(f"{args.seed}-{gen}")
        stats = {"bad": 0}
        work = tmp / gen
        work.mkdir(parents=True, exist_ok=True)

        has_archives = any(p.suffix.lower() == ".zip" for p in work.rglob("*"))
        if not has_archives and not args.no_download:
            kaggle_download(KAGGLE[gen], work)

        process_archive_dir(work, work, dst, n_train, rng, stats, args.keep_tmp)
        if not dst.exists() or not any(dst.rglob("*")):
            # maybe the user put an already-extracted folder into <tmp>/<gen>/
            if not copy_images_from_folder(work, dst, n_train, rng):
                sys.exit(f"No train/val/ai|nature images found for {gen} in {work}")

        counts = count_images(dst)
        log(f"  result: {counts}  corrupt skipped: {stats['bad']}")
        if counts["val/ai"] == 0 or counts["val/nature"] == 0:
            sys.exit(f"{gen}: val images missing, keeping temp files for inspection")
        (dst / ".done").write_text(str(counts))
        if not args.keep_tmp:
            shutil.rmtree(work, ignore_errors=True)
        log(f"  {gen} finished, free space now {free_gb(tmp):.1f} GB")

    log("\nAll done. Next:")
    log(f"  python inspect_genimage.py --root {out} --gens {' '.join(args.gens)}")
    log(f"  python make_manifest.py --root {out} --seen sdv14 biggan --unseen adm glide "
        f"--n-train 10000 --n-val 1000 --n-test -1 --out manifests")


if __name__ == "__main__":
    main()
