# Method A：用裁剪代替缩放的预处理模块（Crop-instead-of-Resize）

> ARIN5201 Project 6 · AI-Generated Image Detection
> 负责人：Thomas · 文档版本 v1（2026-11-04）
> 相关代码：`scripts/train_baseline.py`（`--preprocess crop`）、`scripts/make_bias_testsets.py`、`scripts/compare_results.py`

---

## 1. 一句话概括

我们发现，在 GenImage 上训练的 CLIP 线性探针检测器，主要依赖**分辨率/清晰度捷径**（"糊 = 假"），而不是生成痕迹。Method A 把 CLIP 默认的"缩放到 224"换成"**不缩放、直接裁剪**"，从预处理层面消除这条捷径，并在未见生成器和偏差控制测试集上验证效果。

---

## 2. 与课程 guideline 的对应关系

| Guideline 原文（Section 6） | 本方法的对应内容 |
|---|---|
| 6.4.1 Step 2：Select a base model (pre-trained CNNs or ViTs) | 以 UnivFD（冻结 CLIP ViT + 线性分类器）为起点 |
| 6.4.1 Step 3：Design the training framework, including **data preprocessing** | Method A 本身就是预处理模块的替换 |
| 6.4.1 Step 4：Test on hold-out, unseen, and perturbed data | `test_seen` / `test_unseen_*` / 9 组退化测试 |
| 6.4.3：**Address dataset bias and ensure fair evaluation across image types** | 偏差分析 + 偏差控制测试集（第 3、5 节） |
| 6.4.3：Handle diverse generative methods (GANs, Diffusion Models) | BigGAN（GAN）+ ADM（扩散），后续加入 SD1.4、GLIDE |
| 6.4.3：Maintain robustness against image transformations | JPEG 90/70/50、模糊 σ=0.5/1/2、降采样 75/50/25% |
| 6.2.1 Input：JPEG/PNG with varying sources | 格式对齐测试（真图 JPEG vs 假图 PNG） |

老师的三条建议：
1. **不过度野心**：只替换预处理，不训练大模型；冻结 CLIP + 线性分类器在笔记本 GPU（RTX 5070 Ti, 12GB）上几分钟可跑完。
2. **方案不是由 AI 直接给出**：动机来自我们自己的实验证据（第 3 节）；参考了 SAFE 等论文的做法（第 8 节）。
3. **以开源模型为起点做模块替换**：起点 UnivFD（开源），改进点是预处理模块。

---

## 3. 动机：实验证据

### 3.1 数据格式与尺寸（`inspect_genimage.py`，每个文件夹抽查 500 张）

| 文件夹 | 格式 | 色度采样 | 主要尺寸 |
|---|---|---|---|
| BigGAN 假图 | PNG 100% | 无（PNG） | 128×128 全部 |
| ADM 假图 | PNG 100% | 无（PNG） | 256×256 全部 |
| ImageNet 真图 | JPEG 100% | 4:4:4 约 80–90%，4:2:0 约 7–16% | 500×375、375×500 等 |

→ 真假两类在**文件格式**和**分辨率**上都可以被完美区分，这两点都与"是否由 AI 生成"无关。

### 3.2 Baseline 结果（UnivFD 式：冻结 CLIP ViT-B/32 + 逻辑回归，BigGAN 训练 4000 张）

| 测试集 | Acc | AUC | 假图召回 | 真图误报 |
|---|---|---|---|---|
| BigGAN 测试集（见过） | 99.4 | 1.000 | 99.6% | 0.8% |
| BigGAN：假图转 JPEG 4:4:4 | 98.5 | 0.998 | – | 1.2% |
| BigGAN：真图也缩到 128px | 52.6 | 0.848 | – | **94.9%** |
| BigGAN：模糊 σ=2 | 50.7 | 0.725 | – | **98.7%** |
| **ADM（未见）** | **61.1** | 0.914 | **22.9%** | 0.7% |
| ADM：降采样 50% | 92.1 | 0.991 | **99.9%** | 15.7% |
| ADM：真图缩到 128px（比假图还小） | 17.4 | **0.121** | 23.3% | 88.4% |

### 3.3 结论

1. **格式偏差对这个 baseline 影响很小**：假图改成 JPEG 后准确率几乎不变。CLIP 在缩放到 224 的过程中抹掉了细微的压缩痕迹。
2. **分辨率是主要捷径**：
   - 真图一旦变小或变糊，几乎全部被判为假图（误报 95–99%）；
   - ADM 假图（256px）比 BigGAN（128px）清晰，于是 77% 被判为真图；
   - 把 ADM 降采样后，检测率反而升到 99.9%——**图像质量越差，"检测"越好**；
   - 真图被缩得比假图还小时，AUC 降到 0.12（排序完全反转）。
3. 原因：CLIP 默认把所有图缩放到 224。128px 的 BigGAN 被**放大**（变糊），500px 的真图被**缩小**（变锐），模型学到的是"缩放方向"，而不是生成痕迹。

---

## 4. 方法设计

### 4.1 预处理对比

| | Baseline（resize） | Method A（crop） |
|---|---|---|
| 训练 | 短边缩放到 224 → 中心裁剪 224 | **不缩放**，随机裁剪 S×S |
| 测试 | 同上 | **不缩放**，中心裁剪 S×S |
| 归一化 | CLIP mean/std | CLIP mean/std（相同） |
| 主干 | CLIP ViT（冻结） | 同一个 CLIP ViT（冻结），位置编码插值到 S |
| 分类器 | 逻辑回归，阈值在 val 上按平衡准确率选取 | 相同 |

### 4.2 关键实现细节

- **裁剪尺寸 S = 128**：BigGAN 假图只有 128px，S 不能大于最小的训练图尺寸，否则又要放大。
- **主干用 ViT-B/16**：128px 输入在 B/32 下只有 4×4 = 16 个 patch，B/16 下有 8×8 = 64 个 patch。
- **位置编码插值**：CLIP 预训练于 224px，使用 transformers 的 `interpolate_pos_encoding=True` 接受 128px 输入。
- **小于 S 的图像**：只有这类图会被最小限度地放大（例如 `down25` 测试中 32px 的 BigGAN 图）。脚本会统计数量并写入结果 CSV 的 `upscaled` 列，报告中需注明。
- **训练时多裁剪**（可选 `--train-crops K`）：每张训练图取 K 个随机裁剪，作为数据增强。
- **公平对比**：resize 与 crop 使用同一份 manifest、同一个分类器和阈值选择方式。**唯一的变量是预处理**。

### 4.3 为什么这样能去掉捷径

裁剪不改变像素的尺度：BigGAN 的 128px 图、ADM 的 256px 图、真图的 500px 图，进入模型时都保持原始的像素密度。模型就不能再通过"被放大了多少"来判断真假，只能依靠图像本身的局部统计特征，也就是生成痕迹。

---

## 5. 评估协议

### 5.1 测试集

| 类别 | 测试集 | 来源 |
|---|---|---|
| 留出（见过的生成器） | `test_seen` | BigGAN 官方 val 全部 12000 张 |
| 未见生成器 | `test_unseen_adm`（后续加 GLIDE、SD1.4 等） | 官方 val |
| 偏差控制 | `fake_jpeg444/420`、`real_jpeg420`、`all_jpeg444`、`size_aligned`、`size_jpeg_aligned`、`real_down128` | `make_bias_testsets.py` |
| 退化 | `jpeg90/70/50`、`blur0.5/1/2`、`down75/50/25` | 同上（与 proposal 第 4 节一致） |

`size_aligned` 会把真图对齐到**该生成器假图的尺寸**（BigGAN 128px，ADM 256px，SD 512px）。`real_down128` 把真图统一缩到 128px（对 ADM 来说就是"反向尺寸"探针，对应 3.2 中 AUC 0.121 那一行）。

### 5.2 指标

- 主指标：**AUC**（与阈值无关）、**Balanced Accuracy**
- 同时报告：Acc、Precision、Recall（假图召回）、F1、AP、**真图误报率**
- 阈值只在 val 上选，测试集不做任何调参
- 只含假图的测试集（如 EvalGEN）只报告 Recall/TPR

### 5.3 实验列表

| 编号 | 主干 | 预处理 | 目的 | 结果文件 |
|---|---|---|---|---|
| E0 | ViT-B/32 | resize | 原始 baseline（已完成，见 3.2） | `results_biggan_bias.csv`（BigGAN 部分）、`results_biggan2adm.csv`（ADM 部分） |
| E1 | ViT-B/16 | resize | 控制变量：排除"换主干"带来的影响 | `results_resize_b16.csv` |
| E2 | ViT-B/16 | **crop 128** | **Method A** | `results_crop128_b16.csv` |
| E3 | ViT-B/16 | crop 128, `--train-crops 4` | 多裁剪增强（消融） | `results_crop128x4_b16.csv` |
| E4 | E2/E3 中较好的配置 | seed 0/1/2 | 稳定性：报告均值 ± 标准差 | `results_crop128_b16_s{0,1,2}.csv` |
| E1b | ViT-B/16 | resize + 训练时 JPEG 增强 | 控制变量：JPEG 增强本身的作用 | `results_resize_b16_jpeg.csv` |
| E2b | ViT-B/16 | **crop 128 + 训练时 JPEG 增强** | **Method A 修正版**：裁剪同时去掉格式捷径 | `results_crop128_b16_jpeg.csv` |
| E5（下一阶段） | ViT-L/14 | crop 224 vs resize | SD1.4 训练（GenImage 标准协议），测 BigGAN/ADM/GLIDE 等 | 待定 |

**E1 vs E2 是核心对比**：同一主干，只差预处理。

### 5.4 预期与判断标准

Method A 有效的标志：
- `test_unseen_adm` 的 AUC 和假图召回率提升；
- `size_aligned`、`blur`、`down` 测试中，真图误报率不再接近 100%；
- `test_seen` 可能略有下降（128px 输入特征变弱）。这是可以接受的代价，需要在报告中如实说明。

---

## 6. 运行步骤

```powershell
# 0. 重新生成偏差测试集（size_aligned 已改为按生成器尺寸对齐；已存在的其他测试集会自动跳过）
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_b2a
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_b2a --source test_unseen_adm

# E1：ViT-B/16 + resize
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --out results_resize_b16.csv

# E2：ViT-B/16 + crop 128（Method A）
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --preprocess crop --crop-size 128 --out results_crop128_b16.csv

# E3：多裁剪增强
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_b2a --model openai/clip-vit-base-patch16 --preprocess crop --crop-size 128 --train-crops 4 --out results_crop128x4_b16.csv

# 对比表（AUC 和 Acc 各一张）
python scripts\compare_results.py results_resize_b16.csv results_crop128_b16.csv results_crop128x4_b16.csv --metric auc --out comparison_auc.md
python scripts\compare_results.py results_resize_b16.csv results_crop128_b16.csv results_crop128x4_b16.csv --metric acc --out comparison_acc.md
```

E4 用不同的 `--seed` 重复 E2 或 E3，并改 `--out` 文件名。

---

## 7. 局限与风险（报告中需要说明）

1. **物体尺度可能成为新捷径**：从 500px 真图裁 128 看到的是局部；从 128px BigGAN 图裁 128 看到的是整张图。ADM（256px）的结果比 BigGAN 更能说明问题；E5 用 SD1.4（512px）训练可以进一步缓解。
2. **128px 输入削弱 CLIP 特征**：CLIP 预训练于 224px，`test_seen` 上的准确率可能下降。
3. **小图仍需放大**：例如 `down25` 中的 32px 图。数量记录在 `upscaled` 列。
4. **冻结特征 + 线性分类器**：预处理改变的是输入，特征本身不会为任务调整。如果效果有限，可以把同一个预处理模块接到组里可训练的方法（Effort / LoRA）上。
5. **训练集只有 BigGAN**：这是 BigGAN 上的现象。E5 用 SD1.4 训练，验证结论是否普遍成立。

---

## 8. 参考文献

- **GenImage**：Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023 Datasets & Benchmarks.
- **UnivFD（起点）**：Ojha, Li, Lee, *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023.
- **SAFE（裁剪代替缩放的来源）**：Li et al., *Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective*, KDD 2025. arXiv:2408.06741. 代码：github.com/Ouxiang-Li/SAFE
- **Fake or JPEG?（GenImage 的 JPEG/尺寸偏差）**：Grommelt et al., ECCV 2024 Workshops. arXiv:2403.17608
- **B-Free（对齐数据、无缩放裁剪）**：Guillaro et al., *A Bias-Free Training Paradigm for More General AI-generated Image Detection*, CVPR 2025. 代码：github.com/grip-unina/B-Free
- **Aligned Datasets**：Rajan et al., *Aligned Datasets Improve Detection of Latent Diffusion-Generated Images*, ICLR 2025. arXiv:2410.11835
- **NPR（上采样痕迹）**：Tan et al., *Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection*, CVPR 2024.
- **Effort（组内方法）**：Yan et al., *Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection*, ICML 2025.
- **AIDE**：Yan et al., *A Sanity Check for AI-generated Image Detection*, ICLR 2025. arXiv:2406.19435

---

## 9. 实验记录（每次跑完补充）

| 日期 | 实验 | 配置 | test_seen AUC | ADM AUC | ADM 召回 | 备注 |
|---|---|---|---|---|---|---|
| 2026-11-04 | E0 | B/32 resize | 1.000 | 0.914 | 22.9% | baseline，见 3.2 |
| 2026-11-04 | E1 | B/16 resize | 1.000 | 0.939 | – | 换主干后分辨率捷径明显减弱 |
| 2026-11-04 | E2 | B/16 crop 128 | 0.998 | 0.926 | – | 整体变差，JPEG/模糊测试掉得最多，见第 10 节 |
| 2026-11-04 | E1b | B/16 resize + JPEG 增强 | 0.999 | 0.924 | 6.4% | ADM JPEG 测试明显变好；ADM 上阈值失效（Acc 53%） |
| 2026-11-04 | E2b | B/16 crop 128 + JPEG 增强 | 0.996 | 0.884 | 23.4% | 见过数据上的 JPEG 敏感性修好了，ADM 仍然变差，见第 11 节 |
| | E3 | B/16 crop 128 ×4 | | | | |

---

## 10. E1 vs E2 结果与分析（2026-11-04）

AUC（%），完整表格见 `comparison_auc.md`：

| 测试集 | E1 resize | E2 crop 128 | Δ |
|---|---|---|---|
| 见过的 BigGAN | 100.0 | 99.8 | −0.2 |
| 未见的 ADM | 93.9 | 92.6 | −1.3 |
| BigGAN size_aligned | 93.1 | 78.1 | −15.0 |
| ADM size_aligned | 89.1 | 67.5 | −21.7 |
| ADM 假图转 JPEG 4:4:4 | 92.8 | 86.6 | −6.2 |
| ADM jpeg50 | 92.7 | 69.1 | −23.6 |
| ADM blur1 | 98.0 | 67.5 | −30.5 |
| BigGAN down25 | 96.3 | 33.3 | −63.0 |

**发现 1：主干 patch 大小影响分辨率捷径。** 同为 resize，B/32 → B/16 后 BigGAN size_aligned AUC 从 0.848 升到 0.931，ADM size_aligned 从 0.854 升到 0.891。E1 作为控制变量是必要的。

**发现 2：裁剪去掉了分辨率捷径，却暴露了格式捷径。** resize 会缩小 500px 真图，顺带抹平 JPEG 压缩痕迹；crop 保留原始像素，真图的 JPEG 痕迹完整进入模型。由于真图 100% 是 JPEG、假图 100% 是 PNG，模型转而依赖"有没有 JPEG 痕迹"，所以 JPEG 压缩和模糊测试掉得最多。**预处理方式决定了模型利用哪一个偏差。**

**注意：** crop 模式下，down25、blur 等测试中多数图像小于 128px，只能被放大，与训练分布不同，这几项需要谨慎解读（数量见结果 CSV 的 `upscaled` 列）。

**修正（E2b）：** 训练时对真图和假图都随机做 JPEG 重编码（质量 70–100，4:4:4 / 4:2:0 随机，概率 0.5），让 JPEG 痕迹不再能区分两类。E1b 是同样增强下的 resize 对照。验证的假设：如果格式捷径是 crop 变差的原因，E2b 应明显好于 E2，并接近或超过 E1b。

---

## 11. E1b / E2b 结果与结论（2026-11-04）

AUC（%），完整表格见 `comparison_auc_v2.md`：

| 测试集 | E1 resize | E1b resize+JPEG | E2 crop | E2b crop+JPEG |
|---|---|---|---|---|
| 见过的 BigGAN | 100.0 | 99.9 | 99.8 | 99.6 |
| **未见的 ADM** | **93.9** | 92.4 | 92.6 | 88.4 |
| BigGAN jpeg50 | 99.9 | 99.9 | 92.2 | **98.0** |
| ADM jpeg50 | 92.7 | **96.3** | 69.1 | 77.1 |
| ADM jpeg90 | 89.5 | **96.3** | 87.1 | 87.8 |
| ADM 假图转 JPEG 4:4:4 | 92.8 | **94.9** | 86.6 | 85.8 |
| ADM size_aligned | **89.1** | 87.8 | 67.5 | 62.3 |
| ADM blur1 | **98.0** | 97.9 | 67.5 | 65.2 |

**结论 1：JPEG 增强修好了 crop 在"见过的数据"上的 JPEG 敏感性**（BigGAN jpeg50 92.2 → 98.0），说明第 10 节"crop 暴露了格式捷径"的判断部分成立。

**结论 2：但在未见的 ADM 上，crop 仍然明显落后于 resize**（88–93 vs 92–94；size_aligned 62–68 vs 88–89）。格式捷径只解释了 JPEG 相关测试的一部分差距。剩下的差距更可能来自：
- 裁剪保留了真图"相机原始分辨率"的像素统计（每个像素包含的细节量），模型把它当成"真"的依据。一旦真图被缩小（size_aligned）或变糊，就被判成假图；
- 128px 输入下 CLIP 特征变弱；
- 只用 128px 的 BigGAN 训练，与 SAFE（ProGAN 256px）、B-Free（504px 裁剪）的设定差距很大。

**结论 3：JPEG 增强对 resize 也有用**（E1b 在 ADM 的 JPEG 类测试上普遍 +2 到 +7），这是本轮最稳定的正结果。

**结论 4：阈值跨生成器失效。** ADM 上 AUC 有 0.92，但用 BigGAN val 选的阈值时，Acc 只有 53%（E1b），假图召回 6.4%。报告中应同时给 AUC 和 Acc，并单独讨论阈值迁移问题。

**注意：** 以上都是单一种子。差距在 1–2 个点的结论需要多种子确认；20–30 个点的差距是可靠的。crop 模式下 down25 测试有约 98% 的图像小于 128px 需要放大，该项对 crop 无效。

**当前判断：** 在"BigGAN 训练 + 冻结 CLIP + 128 裁剪"这个设定下，Method A 是负结果。是否放弃 crop，取决于 E5：用 512px 的 SD1.4 训练、224 裁剪，这才是 crop 的适用场景，也是 GenImage 的标准协议。

---

## 12. 多生成器训练（BigGAN+ADM）与 GLIDE 测试（2026-11-04）

设定：`b2g` = 只用 BigGAN 训练（每类 2000 张）；`ba2g` = BigGAN+ADM 训练（每个生成器每类 1000 张，总量相同）。GLIDE（256px PNG，扩散模型）为两组共同的未见生成器。全部为 ViT-B/16 + 训练时 JPEG 增强，单一种子。AUC（%），完整表格见 `comparison_auc_v3.md`：

| GLIDE 测试集 | resize b2g | resize ba2g | crop b2g | crop ba2g |
|---|---|---|---|---|
| 原始 | **99.1** | 98.2 | 91.7 | 97.7 |
| size_aligned（真图对齐到 256） | **98.5** | 97.4 | 69.0 | 85.2 |
| real_down128（真图比假图小） | 36.4 | **59.8** | 23.8 | 50.6 |
| jpeg50 | 99.1 | **99.4** | 75.6 | 88.4 |
| blur1 | **98.0** | 95.2 | 56.1 | 59.7 |
| down25 | **89.9** | 85.5 | 48.7 | 54.0 |

**结论 1：GLIDE 对这个 baseline 不难。** 只用 BigGAN 训练，GLIDE AUC 就有 99.1，而 ADM 只有 92.2。跨生成器难度很依赖目标生成器。

**结论 2：生成器多样性削弱了分辨率捷径，但没有提高 GLIDE 上的 AUC。** resize 下加入 ADM 后，GLIDE 原始 AUC 略降（99.1 → 98.2），但 real_down128 从 36.4 升到 59.8（+23.4），说明模型不再那么依赖"小 = 假"。

**结论 3：crop 从多样性中获益最大，但仍不如 resize。** crop 加入 ADM 后，GLIDE +6.0、size_aligned +16.2、jpeg50 +12.8，说明之前 crop 失败有一部分是因为训练数据只有 128px 的 BigGAN。但它在所有测试上仍低于 resize，在模糊和强降采样下接近随机（AUC 54–60）。可能的解释是：裁剪后的模型依赖高频细节，而模糊恰好去掉了这些细节。

**对 Method A 的最终判断：** 在本项目的数据和算力条件下，"裁剪代替缩放"不如"缩放 + JPEG 增强"，不建议作为最终方法。它作为负结果和分析仍有价值：说明预处理会决定模型利用哪种偏差，以及裁剪类方法对训练数据分辨率和多样性的依赖。

**注意：** GLIDE 在 ba2g 中与训练用的 ADM 同为 256px，size_aligned 一行已对此做了控制。所有结果为单一种子；AUC 在 99 附近的差异（如 99.1 vs 98.2）不足以下结论。
