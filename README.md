# ARIN5201 CV Project — 数据与 baseline 跑通流程

## 0. 安装依赖
```
pip install kaggle pillow scikit-learn transformers torch
```
```
pip install -U kaggle
```
Kaggle token（新版是 KGAT_ 开头的 API Token）：在 PowerShell 里运行
```
mkdir $env:USERPROFILE\.kaggle -Force
Set-Content -Path $env:USERPROFILE\.kaggle\access_token -Value "你的KGAT_token" -NoNewline
```
验证：`kaggle datasets files vtphatt2/GenImage-BigGAN`

## 1. 先只下一个生成器（BigGAN 最小）
```
python scripts\prepare_genimage_laptop.py --tmp data\genimage_tmp --out data\GenImage_small --gens biggan
```

## 2. 检查数据（格式 / 尺寸 / 色度采样）
```
python scripts\inspect_genimage.py --root data\GenImage_small --gens biggan
```

## 3. 生成跑通用的小划分（只有 BigGAN，先不需要 unseen）
```
python scripts\make_manifest.py --root data\GenImage_small --seen biggan --n-train 2000 --n-val 500 --n-test -1 --out manifests_smoke
```

## 4. 跑通 baseline（冻结 CLIP + 线性分类）
```
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_smoke --max-per-split 1000
```
能输出 acc / bacc / auc 等指标并生成 results_baseline.csv，就说明整条流水线通了。
之后把自己的方法替换掉 baseline 的"特征 + 分类器"部分即可。

## 5. 偏差控制 / 退化测试集（在 baseline 饱和时找出它的失败点）
```
python scripts\make_bias_testsets.py --root data\GenImage_small --manifests manifests_smoke
python scripts\train_baseline.py --root data\GenImage_small --manifests manifests_smoke --out results_biggan_bias.csv
```
默认每类抽 2000 张测试图，生成 15 组测试集（格式 / 色度 / 尺寸对齐 + JPEG / 模糊 / 降采样）。
已经提取过的特征会被缓存，第二条命令只会给新的测试集提特征。
跨生成器（ADM 下好并加进 manifest 后）：加 `--source test_unseen_adm`。

## 6. Method A（用裁剪代替缩放）
方法说明、动机证据、实验列表和运行命令见 **METHOD_A.md**。
对比结果：`python scripts\compare_results.py results_resize_b16.csv results_crop128_b16.csv --metric auc`

## 7. 跑通之后再扩展
1. 下载其余生成器：去掉 `--gens biggan` 重跑第 1 步
2. 按全组统一的协议重新生成 manifests（例如 `--seen sdv14 biggan --unseen adm glide`）
3. 做偏差控制测试集（色度互换、统一 JPEG 重编码、统一尺寸）
