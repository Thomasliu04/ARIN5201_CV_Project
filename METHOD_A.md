# Method A: Crop-instead-of-Resize Preprocessing

> ARIN5201 Project 6 · AI-Generated Image Detection
> Owner: Thomas · Document version v2 (2026-11-04)
> Code: `scripts/train_baseline.py` (`--preprocess crop`, `--train-jpeg`), `scripts/make_bias_testsets.py`, `scripts/compare_results.py`

---

## 1. Summary

A CLIP linear-probe detector trained on GenImage relies mainly on a **resolution / sharpness shortcut** ("blurry = fake") rather than on generation artifacts. Method A replaces CLIP's default "resize to 224" with "**no resizing, crop only**" to remove this shortcut at the preprocessing level, and evaluates the effect on unseen generators and on bias-controlled test sets.

**Outcome (see §10–12):** in our setting, cropping does not beat resizing combined with training-time JPEG augmentation. Method A is therefore reported as a negative result, but the experiments show *why*: the preprocessing choice decides which dataset bias the model exploits.

---

## 2. Mapping to the course guidelines

| Guideline (Section 6) | Corresponding work |
|---|---|
| 6.4.1 Step 2: Select a base model (pre-trained CNNs or ViTs) | Starting point: UnivFD (frozen CLIP ViT + linear classifier) |
| 6.4.1 Step 3: Design the training framework, including **data preprocessing** | Method A replaces the preprocessing module |
| 6.4.1 Step 4: Test on hold-out, unseen, and perturbed data | `test_seen` / `test_unseen_*` / 9 degradation sets |
| 6.4.3: **Address dataset bias and ensure fair evaluation across image types** | Bias analysis + bias-controlled test sets (§3, §5) |
| 6.4.3: Handle diverse generative methods (GANs, Diffusion Models) | BigGAN (GAN), ADM and GLIDE (diffusion) |
| 6.4.3: Maintain robustness against image transformations | JPEG 90/70/50, blur σ = 0.5/1/2, downsampling 75/50/25% |
| 6.2.1 Input: JPEG/PNG with varying sources | Format-alignment tests (real JPEG vs fake PNG) |

The instructor's three tips:
1. **Not too ambitious**: only the preprocessing is replaced and no large model is trained; frozen CLIP + a linear classifier runs in minutes on a laptop GPU (RTX 5070 Ti, 12 GB).
2. **The method is not handed over by an AI tool**: the motivation comes from our own experimental evidence (§3); the cropping idea follows published work such as SAFE (§8).
3. **Start from an open-source model and replace a module**: starting point UnivFD (open source); the replaced module is the preprocessing.

---

## 3. Motivation: experimental evidence

### 3.1 File format and size (`inspect_genimage.py`, 500 images sampled per folder)

| Folder | Format | Chroma subsampling | Size |
|---|---|---|---|
| BigGAN fakes | PNG 100% | none (PNG) | all 128×128 |
| ADM fakes | PNG 100% | none (PNG) | all 256×256 |
| GLIDE fakes | PNG 100% | none (PNG) | all 256×256 |
| ImageNet reals | JPEG 100% | 4:4:4 ≈ 80–90%, 4:2:0 ≈ 7–16% | 500×375, 375×500, … |

→ Real and fake images can be perfectly separated by **file format** and by **resolution**, neither of which has anything to do with being AI-generated.

### 3.2 Baseline results (UnivFD-style: frozen CLIP ViT-B/32 + logistic regression, 4000 BigGAN training images)

| Test set | Acc | AUC | Fake recall | Real FPR |
|---|---|---|---|---|
| BigGAN (seen) | 99.4 | 1.000 | 99.6% | 0.8% |
| BigGAN: fakes → JPEG 4:4:4 | 98.5 | 0.998 | – | 1.2% |
| BigGAN: reals also resized to 128 px | 52.6 | 0.848 | – | **94.9%** |
| BigGAN: blur σ = 2 | 50.7 | 0.725 | – | **98.7%** |
| **ADM (unseen)** | **61.1** | 0.914 | **22.9%** | 0.7% |
| ADM: downsampled to 50% | 92.1 | 0.991 | **99.9%** | 15.7% |
| ADM: reals resized to 128 px (smaller than fakes) | 17.4 | **0.121** | 23.3% | 88.4% |

### 3.3 Conclusions

1. **The format bias has little effect on this baseline**: converting fakes to JPEG barely changes accuracy. Resizing to 224 wipes out subtle compression traces.
2. **Resolution is the main shortcut**:
   - once real images become smaller or blurrier, almost all of them are classified as fake (FPR 95–99%);
   - ADM fakes (256 px) are sharper than BigGAN fakes (128 px), so 77% of them are classified as real;
   - downsampling ADM images *raises* recall to 99.9% — **the worse the image quality, the better the "detection"**;
   - when reals are made smaller than the fakes, AUC drops to 0.12 (the ranking is fully inverted).
3. Cause: CLIP resizes every image to 224. 128 px BigGAN fakes are **upscaled** (blurred) and ~500 px reals are **downscaled** (sharpened), so the model learns the direction of resizing, not generation artifacts.

---

## 4. Method design

### 4.1 Preprocessing comparison

| | Baseline (resize) | Method A (crop) |
|---|---|---|
| Training | resize short side to 224 → center crop 224 | **no resizing**, random S×S crop |
| Testing | same as above | **no resizing**, center S×S crop |
| Normalization | CLIP mean/std | CLIP mean/std (same) |
| Backbone | frozen CLIP ViT | same frozen CLIP ViT, position embeddings interpolated to S |
| Classifier | logistic regression, threshold chosen on val by balanced accuracy | same |

### 4.2 Implementation details

- **Crop size S = 128**: BigGAN fakes are only 128 px, so S cannot exceed the smallest training image without upscaling again.
- **ViT-B/16 backbone**: a 128 px input gives only 4×4 = 16 patches with B/32, but 8×8 = 64 patches with B/16.
- **Position-embedding interpolation**: CLIP is pre-trained at 224 px; `interpolate_pos_encoding=True` in transformers accepts 128 px inputs.
- **Images smaller than S**: only these are minimally upscaled (e.g. 32 px BigGAN images in the `down25` test). The script counts them and writes the count to the `upscaled` column of the result CSV; the report must mention it.
- **Multiple training crops** (optional `--train-crops K`): K random crops per training image as data augmentation.
- **Training-time JPEG augmentation** (optional `--train-jpeg QMIN QMAX`, added in §10): training images of *both* classes are re-encoded as JPEG with random quality in [QMIN, QMAX] and random 4:4:4 / 4:2:0 subsampling, with probability `--train-jpeg-p` (default 0.5). Val and test images are never touched.
- **Fair comparison**: resize and crop use the same manifests, classifier and threshold selection. **The only variable is the preprocessing.**

### 4.3 Why this should remove the shortcut

Cropping does not change the pixel scale: 128 px BigGAN images, 256 px ADM images and ~500 px real images all enter the model at their native pixel density. The model can no longer tell real from fake by how much an image was rescaled, and has to rely on local image statistics, i.e. generation artifacts.

---

## 5. Evaluation protocol

### 5.1 Test sets

| Category | Test set | Source |
|---|---|---|
| Hold-out (seen generator) | `test_seen` | all 12,000 images of the official BigGAN val split |
| Unseen generators | `test_unseen_adm`, `test_unseen_glide` | official val splits |
| Bias-controlled | `fake_jpeg444/420`, `real_jpeg420`, `all_jpeg444`, `size_aligned`, `size_jpeg_aligned`, `real_down128` | `make_bias_testsets.py` |
| Degradations | `jpeg90/70/50`, `blur0.5/1/2`, `down75/50/25` | same script (matches Section 4 of the proposal) |

`size_aligned` resizes reals to **the fake size of that generator** (BigGAN 128 px, ADM/GLIDE 256 px, SD 512 px). `real_down128` resizes reals to 128 px for every generator; for ADM/GLIDE this makes reals *smaller* than fakes, a "reversed size" probe (the AUC 0.121 row in §3.2).

### 5.2 Metrics

- Primary: **AUC** (threshold-free) and **balanced accuracy**
- Also reported: accuracy, precision, recall (fake recall), F1, AP, **real false-positive rate**
- The threshold is chosen on val only; no tuning on any test set
- Fake-only test sets (e.g. EvalGEN) report recall / TPR only

### 5.3 Experiment list

| ID | Backbone | Preprocessing | Purpose | Result file | Status |
|---|---|---|---|---|---|
| E0 | ViT-B/32 | resize | original baseline (§3.2) | `results_biggan_bias.csv` (BigGAN), `results_biggan2adm.csv` (ADM) | done |
| E1 | ViT-B/16 | resize | control: effect of the backbone change | `results_resize_b16.csv` | done |
| E2 | ViT-B/16 | **crop 128** | **Method A** | `results_crop128_b16.csv` | done |
| E1b | ViT-B/16 | resize + JPEG aug. | control: effect of JPEG augmentation alone | `results_resize_b16_jpeg.csv` | done |
| E2b | ViT-B/16 | **crop 128 + JPEG aug.** | **revised Method A**: crop and remove the format shortcut | `results_crop128_b16_jpeg.csv` | done |
| G1–G4 | ViT-B/16 | resize / crop, + JPEG aug. | BigGAN vs BigGAN+ADM training, GLIDE as unseen | `r_b2g_*.csv`, `r_ba2g_*.csv` | done |
| E3 | ViT-B/16 | crop 128, `--train-crops 4` | multi-crop augmentation (ablation) | `results_crop128x4_b16.csv` | not run |
| E4 | best configuration | seeds 0/1/2 | stability: mean ± std | – | not run |
| E5 | ViT-L/14 | crop 224 vs resize | SD1.4 training (GenImage standard protocol) | – | not run |

**E1 vs E2 is the core comparison**: same backbone, only the preprocessing differs.

### 5.4 Success criteria

Method A works if:
- AUC and fake recall on unseen generators improve;
- the real FPR on `size_aligned`, `blur` and `down` tests is no longer close to 100%;
- `test_seen` may drop slightly (weaker features at 128 px). This is an acceptable cost and must be reported.

---

## 6. How to run

```powershell
# bias-controlled test sets (already existing variants are skipped)
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_b2a
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_b2a --source test_unseen_adm

# E1: ViT-B/16 + resize
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --out results_resize_b16.csv
# E2: ViT-B/16 + crop 128 (Method A)
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --preprocess crop --crop-size 128 --out results_crop128_b16.csv
# E1b / E2b: + training-time JPEG augmentation
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --train-jpeg 70 100 --out results_resize_b16_jpeg.csv
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --preprocess crop --crop-size 128 --train-jpeg 70 100 --out results_crop128_b16_jpeg.csv

# comparison table
python scripts\compare_results.py results_resize_b16.csv results_resize_b16_jpeg.csv results_crop128_b16.csv results_crop128_b16_jpeg.csv --metric auc --out comparison_auc_v2.md
```

Multi-generator experiment (§12):

```powershell
python scripts\make_manifest.py --root data\GenImage_small --seen biggan --unseen adm glide --n-train 2000 --n-val 500 --n-test -1 --out manifests_b2g
python scripts\make_manifest.py --root data\GenImage_small --seen biggan adm --unseen glide --n-train 1000 --n-val 250 --n-test -1 --out manifests_ba2g
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_b2g --source test_unseen_glide
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_ba2g --source test_unseen_glide
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2g --model openai/clip-vit-base-patch16 --train-jpeg 70 100 --out r_b2g_resize_jpeg.csv
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_ba2g --model openai/clip-vit-base-patch16 --train-jpeg 70 100 --out r_ba2g_resize_jpeg.csv
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2g --model openai/clip-vit-base-patch16 --preprocess crop --crop-size 128 --train-jpeg 70 100 --out r_b2g_crop_jpeg.csv
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_ba2g --model openai/clip-vit-base-patch16 --preprocess crop --crop-size 128 --train-jpeg 70 100 --out r_ba2g_crop_jpeg.csv
python scripts\compare_results.py r_b2g_resize_jpeg.csv r_ba2g_resize_jpeg.csv r_b2g_crop_jpeg.csv r_ba2g_crop_jpeg.csv --metric auc --out comparison_auc_v3.md
```

For E4, repeat a configuration with different `--seed` values and different `--out` file names.

---

## 7. Limitations and risks

1. **Object scale may become a new shortcut**: a 128 px crop of a ~500 px real image shows a local region, while a 128 px BigGAN image shows the whole scene. Results on ADM/GLIDE (256 px) are more informative than on BigGAN; training on SD1.4 (512 px, E5) would reduce this further.
2. **128 px inputs weaken CLIP features**: CLIP is pre-trained at 224 px, so accuracy on `test_seen` may drop.
3. **Small images still need upscaling**: e.g. 32 px images in `down25`; counts are in the `upscaled` column.
4. **Frozen features + linear classifier**: preprocessing changes the input, but the features are not adapted to the task. The same preprocessing module could be plugged into a trainable method (Effort / LoRA).
5. **Single seed**: differences of 1–2 points need multiple seeds to be trusted; differences of 20–30 points are reliable.

---

## 8. References

- **GenImage**: Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023 Datasets & Benchmarks.
- **UnivFD (starting point)**: Ojha, Li, Lee, *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023.
- **SAFE (source of crop-instead-of-resize)**: Li et al., *Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective*, KDD 2025. arXiv:2408.06741. Code: github.com/Ouxiang-Li/SAFE
- **Fake or JPEG? (JPEG / size biases in GenImage)**: Grommelt et al., ECCV 2024 Workshops. arXiv:2403.17608
- **B-Free (aligned data, crop without resizing)**: Guillaro et al., *A Bias-Free Training Paradigm for More General AI-generated Image Detection*, CVPR 2025. Code: github.com/grip-unina/B-Free
- **Aligned Datasets**: Rajan et al., *Aligned Datasets Improve Detection of Latent Diffusion-Generated Images*, ICLR 2025. arXiv:2410.11835
- **NPR (up-sampling artifacts)**: Tan et al., *Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection*, CVPR 2024.
- **Effort (used in our group)**: Yan et al., *Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection*, ICML 2025.
- **AIDE**: Yan et al., *A Sanity Check for AI-generated Image Detection*, ICLR 2025. arXiv:2406.19435

---

## 9. Experiment log

| Date | Exp. | Configuration | test_seen AUC | Unseen AUC | Unseen recall | Notes |
|---|---|---|---|---|---|---|
| 2026-11-04 | E0 | B/32 resize, BigGAN train | 1.000 | ADM 0.914 | 22.9% | baseline, §3.2 |
| 2026-11-04 | E1 | B/16 resize | 1.000 | ADM 0.939 | – | resolution shortcut clearly weaker with B/16 |
| 2026-11-04 | E2 | B/16 crop 128 | 0.998 | ADM 0.926 | – | worse overall, largest drops on JPEG/blur tests, §10 |
| 2026-11-04 | E1b | B/16 resize + JPEG aug. | 0.999 | ADM 0.924 | 6.4% | ADM JPEG tests improve; threshold fails on ADM (Acc 53%) |
| 2026-11-04 | E2b | B/16 crop 128 + JPEG aug. | 0.996 | ADM 0.884 | 23.4% | JPEG sensitivity fixed on seen data, ADM still worse, §11 |
| 2026-11-04 | G1 | B/16 resize + JPEG, BigGAN | 1.000 | GLIDE 0.991 | – | §12 |
| 2026-11-04 | G2 | B/16 resize + JPEG, BigGAN+ADM | 0.996 | GLIDE 0.982 | – | resolution shortcut weaker (`real_down128` +23.4) |
| 2026-11-04 | G3 | B/16 crop + JPEG, BigGAN | 0.996 | GLIDE 0.917 | – | |
| 2026-11-04 | G4 | B/16 crop + JPEG, BigGAN+ADM | 0.983 | GLIDE 0.977 | – | crop gains most from generator diversity |

`test_seen` in G2/G4 contains BigGAN and ADM, so it is not directly comparable with G1/G3.

---

## 10. E1 vs E2: results and analysis

AUC (%); full table in `comparison_auc.md`:

| Test set | E1 resize | E2 crop 128 | Δ |
|---|---|---|---|
| BigGAN (seen) | 100.0 | 99.8 | −0.2 |
| ADM (unseen) | 93.9 | 92.6 | −1.3 |
| BigGAN size_aligned | 93.1 | 78.1 | −15.0 |
| ADM size_aligned | 89.1 | 67.5 | −21.7 |
| ADM fakes → JPEG 4:4:4 | 92.8 | 86.6 | −6.2 |
| ADM jpeg50 | 92.7 | 69.1 | −23.6 |
| ADM blur1 | 98.0 | 67.5 | −30.5 |
| BigGAN down25 | 96.3 | 33.3 | −63.0 |

**Finding 1: the backbone patch size affects the resolution shortcut.** With resize in both cases, going from B/32 to B/16 raises BigGAN size_aligned AUC from 0.848 to 0.931 and ADM size_aligned from 0.854 to 0.891. E1 as a control is therefore necessary.

**Finding 2: cropping removes the resolution shortcut but exposes the format shortcut.** Resizing shrinks ~500 px reals and smooths out their JPEG block artifacts; cropping keeps native pixels, so the JPEG traces of reals reach the model intact. Because reals are 100% JPEG and fakes 100% PNG, the model switches to "has JPEG traces = real", which explains why JPEG compression and blur tests drop the most. **The preprocessing choice decides which bias the model exploits.**

**Caution:** in crop mode, most images in `down25` (and some in other degradation tests) are smaller than 128 px and must be upscaled, unlike in training; interpret these rows with care (counts in the `upscaled` column).

**Fix (E2b):** re-encode training images of both classes as JPEG at random (quality 70–100, random 4:4:4 / 4:2:0, probability 0.5) so that JPEG traces no longer separate the classes. E1b is the resize control with the same augmentation. Hypothesis: if the format shortcut caused the drop, E2b should clearly beat E2 and approach or exceed E1b.

---

## 11. E1b / E2b: results and conclusions

AUC (%); full table in `comparison_auc_v2.md`:

| Test set | E1 resize | E1b resize+JPEG | E2 crop | E2b crop+JPEG |
|---|---|---|---|---|
| BigGAN (seen) | 100.0 | 99.9 | 99.8 | 99.6 |
| **ADM (unseen)** | **93.9** | 92.4 | 92.6 | 88.4 |
| BigGAN jpeg50 | 99.9 | 99.9 | 92.2 | **98.0** |
| ADM jpeg50 | 92.7 | **96.3** | 69.1 | 77.1 |
| ADM jpeg90 | 89.5 | **96.3** | 87.1 | 87.8 |
| ADM fakes → JPEG 4:4:4 | 92.8 | **94.9** | 86.6 | 85.8 |
| ADM size_aligned | **89.1** | 87.8 | 67.5 | 62.3 |
| ADM blur1 | **98.0** | 97.9 | 67.5 | 65.2 |

**Conclusion 1: JPEG augmentation fixes crop's JPEG sensitivity on seen data** (BigGAN jpeg50 92.2 → 98.0), partly confirming Finding 2 of §10.

**Conclusion 2: on unseen ADM, crop still lags clearly behind resize** (88–93 vs 92–94; size_aligned 62–68 vs 88–89). The format shortcut explains only part of the gap on JPEG-related tests. The rest is more likely due to:
- cropping preserves the camera-resolution pixel statistics of real photos (amount of detail per pixel), which the model treats as evidence of "real"; once reals are downscaled (size_aligned) or blurred, they are classified as fake;
- weaker CLIP features at 128 px;
- training only on 128 px BigGAN, far from the settings of SAFE (ProGAN, 256 px) and B-Free (504 px crops).

**Conclusion 3: JPEG augmentation also helps resize** (E1b gains +2 to +7 on ADM JPEG-related tests). This is the most stable positive result of this round.

**Conclusion 4: thresholds do not transfer across generators.** On ADM the AUC is 0.92, but with the threshold chosen on BigGAN val, accuracy is only 53% (E1b) and fake recall 6.4%. The report must give both AUC and accuracy and discuss threshold transfer separately.

**Caution:** single seed. Differences of 1–2 points need multiple seeds; differences of 20–30 points are reliable. In crop mode about 98% of `down25` images are smaller than 128 px and must be upscaled, so that row is not valid for crop.

---

## 12. Multi-generator training (BigGAN+ADM) tested on GLIDE

Setup: `b2g` = trained on BigGAN only (2000 images per class); `ba2g` = trained on BigGAN+ADM (1000 per class per generator, same total). GLIDE (256 px PNG, diffusion) is the common unseen generator. All runs use ViT-B/16 + training-time JPEG augmentation, single seed. AUC (%); full table in `comparison_auc_v3.md`:

| GLIDE test set | resize b2g | resize ba2g | crop b2g | crop ba2g |
|---|---|---|---|---|
| clean | **99.1** | 98.2 | 91.7 | 97.7 |
| size_aligned (reals → 256 px) | **98.5** | 97.4 | 69.0 | 85.2 |
| real_down128 (reals smaller than fakes) | 36.4 | **59.8** | 23.8 | 50.6 |
| jpeg50 | 99.1 | **99.4** | 75.6 | 88.4 |
| blur1 | **98.0** | 95.2 | 56.1 | 59.7 |
| down25 | **89.9** | 85.5 | 48.7 | 54.0 |

**Conclusion 1: GLIDE is easy for this baseline.** Trained on BigGAN only, GLIDE AUC is already 99.1, versus 92.2 on ADM. Cross-generator difficulty depends strongly on the target generator.

**Conclusion 2: generator diversity weakens the resolution shortcut but does not raise GLIDE AUC.** With resize, adding ADM slightly lowers clean GLIDE AUC (99.1 → 98.2, within single-seed noise), but `real_down128` rises from 36.4 to 59.8 (+23.4): the model relies much less on "small = fake".

**Conclusion 3: crop benefits most from diversity, but still trails resize.** Adding ADM gives crop +6.0 on GLIDE, +16.2 on size_aligned and +12.8 on jpeg50, so part of crop's earlier failure came from training only on 128 px BigGAN. It nevertheless stays below resize on every test and is close to chance under blur and strong downsampling (AUC 54–60). A likely explanation is that the cropped model depends on high-frequency detail, which blur removes.

**Final judgement on Method A:** with the data and compute available in this project, crop-instead-of-resize is worse than resize + JPEG augmentation and is not recommended as the final method. It remains valuable as a negative result: it shows that preprocessing determines which bias a detector exploits, and that crop-based methods depend on the resolution and diversity of the training data.

**Note:** in `ba2g`, GLIDE has the same 256 px size as the ADM training fakes; the size_aligned row controls for this. All results are single-seed; differences around AUC 99 (e.g. 99.1 vs 98.2) are not conclusive.
