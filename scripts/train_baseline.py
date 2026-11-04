"""
Frozen CLIP image encoder + logistic regression (UnivFD-style linear probe),
with a switchable preprocessing module (Method A, see METHOD_A.md):

  --preprocess resize  (default CLIP pipeline: resize short side to 224 + center crop 224)
  --preprocess crop    (Method A: NO resizing; random crop for train, center crop for
                        val/test, at --crop-size; CLIP position embeddings are interpolated
                        when crop-size != 224)

Everything else (data, classifier, threshold selection, metrics) is identical between the
two modes, so differences in results come from the preprocessing alone.

Usage:
    # baseline (what we ran before)
    python scripts\\train_baseline.py --root data\\GenImage_small --manifests manifests_b2a --out results_resize_b32.csv
    # same backbone as Method A, but resize  (controls for the backbone change)
    python scripts\\train_baseline.py --root data\\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --out results_resize_b16.csv
    # Method A
    python scripts\\train_baseline.py --root data\\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --preprocess crop --crop-size 128 --out results_crop128_b16.csv

Options:
    --train-crops K   K random crops per training image (crop mode only, data augmentation)
    --train-jpeg QMIN QMAX   re-encode TRAINING images (real and fake alike) as JPEG with a random
                      quality in [QMIN, QMAX] and random 4:4:4 / 4:2:0 subsampling, with probability
                      --train-jpeg-p (default 0.5). Removes "has JPEG traces = real" as a shortcut.
                      Works in both resize and crop mode; val/test images are never touched.
    --seed S          seed for random crops (repeat with several seeds and report mean +- std)
    --max-per-split N subsample each split (balanced) for a quick run; 0 = all
Install: pip install torch transformers scikit-learn pillow
"""
import argparse
import csv
import hashlib
import io
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


# ------------------------------------------------------------------ data
def read_manifest(path, max_n, seed=0):
    with open(path, newline="") as f:
        rows = [(r["path"], int(r["label"]), r["generator"]) for r in csv.DictReader(f)]
    if 0 < max_n < len(rows):  # keep classes balanced when subsampling
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rows))
        pos = [i for i in idx if rows[i][1] == 1][: max_n // 2]
        neg = [i for i in idx if rows[i][1] == 0][: max_n - max_n // 2]
        rows = [rows[i] for i in sorted(pos + neg)]
    return rows


def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]


# ------------------------------------------------------------------ Method A: crop module
class CropPreprocess:
    """No-resize preprocessing. Images smaller than the crop are the only ones that get
    (minimally) upscaled; they are counted so the report can state how many."""

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.upscaled = 0

    def __call__(self, im, rng=None):
        s = self.size
        w, h = im.size
        if min(w, h) < s:
            k = s / min(w, h)
            im = im.resize((max(s, round(w * k)), max(s, round(h * k))), Image.BICUBIC)
            self.upscaled += 1
            w, h = im.size
        if rng is not None:  # training: random crop
            left, top = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
        else:                # val / test: center crop
            left, top = (w - s) // 2, (h - s) // 2
        x = np.asarray(im.crop((left, top, left + s, top + s)), dtype=np.float32) / 255.0
        return ((x - self.mean) / self.std).transpose(2, 0, 1)


def jpeg_aug(im, rng, qmin, qmax):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(qmin, qmax + 1)), subsampling=int(rng.choice([0, 2])))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def extract(rows, root, model, proc, args, device, train):
    crop = args.preprocess == "crop"
    if crop:
        pre = CropPreprocess(args.crop_size, proc.image_mean, proc.image_std)
        rng = np.random.default_rng(args.seed) if train else None
        n_rep = args.train_crops if train else 1
    else:
        rng, n_rep = None, 1
    use_jpeg = train and args.train_jpeg is not None
    jrng = np.random.default_rng(args.seed + 1000)
    feats, labels = [], []
    batch_x, batch_y = [], []

    def flush():
        if not batch_x:
            return
        if crop:
            x = torch.from_numpy(np.stack(batch_x)).to(device)
            kw = {"interpolate_pos_encoding": True} if args.crop_size != 224 else {}
        else:
            x = proc(images=batch_x, return_tensors="pt")["pixel_values"].to(device)
            kw = {}
        f = model(pixel_values=x, **kw).image_embeds
        feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
        labels.extend(batch_y)
        batch_x.clear()
        batch_y.clear()

    for i, (p, label, _) in enumerate(rows, 1):
        with Image.open(Path(root) / p) as im:
            im = im.convert("RGB")
            for _ in range(n_rep):
                x = im
                if use_jpeg and jrng.random() < args.train_jpeg_p:
                    x = jpeg_aug(im, jrng, *args.train_jpeg)
                batch_x.append(pre(x, rng) if crop else x.copy())
                batch_y.append(label)
        if len(batch_x) >= args.batch:
            flush()
        if i % 500 == 0 or i == len(rows):
            print(f"    {i}/{len(rows)}", end="\r", flush=True)
    flush()
    print()
    upscaled = pre.upscaled if crop else 0
    if upscaled:
        print(f"    note: {upscaled} images were smaller than {args.crop_size}px and had to be upscaled")
    return np.concatenate(feats), np.asarray(labels), upscaled


# ------------------------------------------------------------------ metrics
def balanced_acc(y, pred):
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return np.nanmean([tpr, tnr])


def best_threshold(y, p):
    """Threshold maximising balanced accuracy on validation."""
    best_t, best_b = 0.5, -1
    for t in np.unique(np.round(p, 4)):
        b = balanced_acc(y, p >= t)
        if b > best_b:
            best_t, best_b = t, b
    return float(best_t)


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    both = len(np.unique(y)) == 2
    return {
        "acc": (tp + tn) / len(y), "bacc": balanced_acc(y, pred),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "auc": roc_auc_score(y, p) if both else float("nan"),
        "ap": average_precision_score(y, p) if both else float("nan"),
        "real_fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "n": len(y),
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root used by make_manifest.py")
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32",
                    help="HuggingFace CLIP id, e.g. openai/clip-vit-base-patch16, openai/clip-vit-large-patch14")
    ap.add_argument("--preprocess", choices=["resize", "crop"], default="resize")
    ap.add_argument("--crop-size", type=int, default=128, help="crop mode only; multiple of the patch size")
    ap.add_argument("--train-crops", type=int, default=1, help="random crops per training image (crop mode)")
    ap.add_argument("--train-jpeg", type=int, nargs=2, metavar=("QMIN", "QMAX"), default=None,
                    help="random JPEG re-encoding of training images (both classes)")
    ap.add_argument("--train-jpeg-p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-per-split", type=int, default=0, help="0 = use everything")
    ap.add_argument("--out", default="results_baseline.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = (f"{args.preprocess}" + (f"{args.crop_size}_x{args.train_crops}" if args.preprocess == "crop" else "")
           + (f"_jpeg{args.train_jpeg[0]}-{args.train_jpeg[1]}p{args.train_jpeg_p}" if args.train_jpeg else "")
           + f"_seed{args.seed}_max{args.max_per_split}")
    print(f"device: {device}   model: {args.model}   preprocess: {cfg}")
    proc = CLIPImageProcessor.from_pretrained(args.model)
    model = CLIPVisionModelWithProjection.from_pretrained(args.model).to(device).eval()
    if args.preprocess == "crop" and args.crop_size % model.config.patch_size:
        raise SystemExit(f"--crop-size must be a multiple of the patch size {model.config.patch_size}")

    mdir = Path(args.manifests)
    cache = mdir / f"feats_{re.sub(r'[^A-Za-z0-9]+', '_', args.model)}"
    cache.mkdir(exist_ok=True)
    upscaled_total = {}

    def load(name, train=False):
        csv_path = mdir / f"{name}.csv"
        rows = read_manifest(csv_path, args.max_per_split)
        f = cache / f"{name}_{file_hash(csv_path)}_{cfg}.npz"  # hash: cache is redone if the csv changes
        if f.exists():
            d = np.load(f)
            X, y, up = d["X"], d["y"], int(d["up"])
        else:
            print(f"  extracting {name} ({len(rows)} images)")
            X, y, up = extract(rows, args.root, model, proc, args, device, train)
            np.savez(f, X=X, y=y, up=up)
        upscaled_total[name] = up
        return X, y

    Xtr, ytr = load("train", train=True)
    Xva, yva = load("val")
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
    t = best_threshold(yva, clf.predict_proba(Xva)[:, 1])
    print(f"threshold chosen on val: {t:.4f}")

    tests = sorted(p.stem for p in mdir.glob("test_*.csv"))
    results = []
    for name in ["val"] + tests:
        X, y = (Xva, yva) if name == "val" else load(name)
        m = metrics(y, clf.predict_proba(X)[:, 1], t)
        results.append({"set": name, **m, "upscaled": upscaled_total.get(name, 0),
                        "model": args.model, "preprocess": args.preprocess,
                        "crop_size": args.crop_size if args.preprocess == "crop" else "",
                        "train_crops": args.train_crops if args.preprocess == "crop" else "",
                        "train_jpeg": f"{args.train_jpeg[0]}-{args.train_jpeg[1]}@{args.train_jpeg_p}" if args.train_jpeg else "",
                        "seed": args.seed})
        print(f"{name:34s} acc={m['acc']:.3f} bacc={m['bacc']:.3f} auc={m['auc']:.3f} "
              f"recall={m['recall']:.3f} real_fpr={m['real_fpr']:.3f} n={m['n']}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
