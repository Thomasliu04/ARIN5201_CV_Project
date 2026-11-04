# ARIN5201 CV Project — Bias-Aware Evaluation for AI-Generated Image Detection

Part of the ARIN5201 / MAIE5421 group project (Project 6: AI-Generated Image Detection, HKUST, Fall 2026).
This repository contains Thomas's part: a data pipeline for GenImage, a suite of **bias-controlled and degraded test sets**, a frozen-CLIP linear-probe baseline (UnivFD-style), and the **Method A** experiments (crop-instead-of-resize preprocessing).

Full method description, evidence and results: **[METHOD_A.md](METHOD_A.md)**.

## Key findings (so far)

- **GenImage has two perfect shortcuts.** Real images are 100% JPEG (mostly 4:4:4 chroma) at ~500×375; fakes are 100% PNG at a fixed size (BigGAN 128×128, ADM/GLIDE 256×256).
- **The baseline learns "blurry = fake".** Trained on BigGAN, a frozen CLIP ViT-B/32 + logistic regression reaches 99.4% accuracy on BigGAN, but only 61.1% on unseen ADM. Blurring or downscaling ADM images *raises* detection recall to 99.9%.
- **Preprocessing decides which bias the model exploits.** Resizing hides JPEG traces but creates a resolution shortcut; cropping removes the resolution cue but exposes the JPEG-format cue.
- **Backbone, JPEG augmentation and generator diversity matter.** ViT-B/16 is less resolution-sensitive than ViT-B/32; training-time JPEG augmentation improves robustness to compression; training on BigGAN+ADM weakens the resolution shortcut.
- **Method A (crop-instead-of-resize) is a negative result in this setting.** It does not beat resize + JPEG augmentation, but the analysis explains why (see METHOD_A.md §10–12).
- **Thresholds do not transfer across generators.** AUC on unseen generators stays high (~0.92) while accuracy with the validation threshold can drop to ~53%; always report AUC and accuracy together.

All results are single-seed unless stated otherwise.

## Repository structure

```
scripts/
  prepare_genimage_laptop.py   download GenImage subsets from the Kaggle mirror, keep only a sampled subset
  inspect_genimage.py          file format, size, JPEG chroma subsampling, corrupt-file check
  make_manifest.py             fixed train/val/test splits (fixed seed, MD5 de-duplication)
  make_bias_testsets.py        bias-controlled and degraded test sets (15 variants)
  train_baseline.py            frozen CLIP + logistic regression; --preprocess resize|crop, --train-jpeg
  compare_results.py           side-by-side Markdown comparison of result CSVs
manifests_*/                   split CSVs (paths relative to the data root; images not included)
results_*.csv, r_*.csv         per-test-set metrics of each experiment
comparison_*.md                comparison tables
METHOD_A.md                    method, evidence, experiment log and analysis
```

Images and cached CLIP features are **not** included (see `.gitignore`). GenImage is licensed CC BY-NC-SA 4.0; download it yourself with the scripts below.

## Setup

Tested on Windows 11, Python 3.12, PyTorch 2.11 + CUDA 12.8 (RTX 5070 Ti Laptop, 12 GB).

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers scikit-learn pillow kaggle
```

RTX 50-series (Blackwell) GPUs need a CUDA 12.8+ build of PyTorch.

Kaggle API token (new `KGAT_...` tokens):

```powershell
mkdir $env:USERPROFILE\.kaggle -Force
Set-Content -Path $env:USERPROFILE\.kaggle\access_token -Value "<your KGAT_ token>" -NoNewline
kaggle datasets files vtphatt2/GenImage-BigGAN   # check that the token works
```

## Pipeline

**1. Download data** (one generator at a time; the archive is deleted after extracting the sampled subset)

```powershell
python scripts\prepare_genimage_laptop.py --tmp data\genimage_tmp --out data\GenImage_small --gens biggan
python scripts\prepare_genimage_laptop.py --tmp data\genimage_tmp --out data\GenImage_small --gens adm glide --train-gens biggan adm
```

`--train-gens` lists the generators whose train split is kept; the others keep only their val split (test-only).

**2. Inspect the data**

```powershell
python scripts\inspect_genimage.py --root data\GenImage_small --gens biggan adm glide
```

**3. Build splits.** `--n-train` / `--n-val` are per class per generator; `--n-test -1` uses the full official val split.

```powershell
python scripts\make_manifest.py --root data\GenImage_small --seen biggan --unseen adm --n-train 2000 --n-val 500 --n-test -1 --out manifests_b2a
```

**4. Build bias-controlled and degraded test sets**

```powershell
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_b2a
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_b2a --source test_unseen_adm
```

| Group | Variants |
|---|---|
| Format / chroma | `fake_jpeg444`, `fake_jpeg420`, `real_jpeg420`, `all_jpeg444` |
| Size | `size_aligned` (reals resized to the generator's fake size), `size_jpeg_aligned`, `real_down128` |
| Degradations | `jpeg90/70/50`, `blur0.5/1/2`, `down75/50/25` |

**5. Train and evaluate.** Every `test_*.csv` in the manifest folder is evaluated; the threshold is chosen on val only.

```powershell
# baseline: ViT-B/16 + resize
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --out results_resize_b16.csv
# + training-time JPEG augmentation (both classes)
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --train-jpeg 70 100 --out results_resize_b16_jpeg.csv
# Method A: crop instead of resize
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --preprocess crop --crop-size 128 --train-jpeg 70 100 --out results_crop128_b16_jpeg.csv
```

**6. Compare**

```powershell
python scripts\compare_results.py results_resize_b16.csv results_resize_b16_jpeg.csv results_crop128_b16_jpeg.csv --metric auc
```

## Experiments in this repository

| Files | Training data | Unseen test | What it compares |
|---|---|---|---|
| `results_biggan_bias.csv`, `results_biggan2adm.csv` | BigGAN | ADM | E0: ViT-B/32 + resize baseline |
| `results_resize_b16.csv`, `results_crop128_b16.csv`, `comparison_auc.md` | BigGAN | ADM | E1 vs E2: resize vs crop (ViT-B/16) |
| `results_*_jpeg.csv`, `comparison_auc_v2.md` | BigGAN | ADM | E1b vs E2b: + JPEG augmentation |
| `r_b2g_*.csv`, `r_ba2g_*.csv`, `comparison_auc_v3.md` | BigGAN vs BigGAN+ADM (same total size) | GLIDE | generator diversity × resize/crop |

## References

- Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023 D&B.
- Ojha, Li, Lee, *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (UnivFD), CVPR 2023.
- Li et al., *Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective* (SAFE), KDD 2025.
- Grommelt et al., *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*, ECCV 2024 Workshops.
- Guillaro et al., *A Bias-Free Training Paradigm for More General AI-generated Image Detection* (B-Free), CVPR 2025.

More in [METHOD_A.md](METHOD_A.md#8-references).
