# 数据准备与标签规范

**首次 clone 请先看 [数据集下载与准备指南](VIDEO_DATASETS_CN.md)**：包含 Mus-V 的 Google Drive 下载链接、其他公共来源的直接下载入口、可复制命令、目录结构、空间预算与标注准备说明。

本页说明下载后的预处理、标签映射与训练缓存。原始数据和人工标签不随 Git 分发；PMC 官方包没有分割 mask，本项目人工标注也不会随 clone 自动获得。标注有效数量以对应 metadata CSV 为准，历史报告中的样本数属于当时的实验快照。

[English](README.md) · [网页标注](collection/README_WEB_CN.md) · [单帧结果](../results/unet/README_CN.md) · [时序结果](../results/temporal/RESULTS_CN.md)

## 目录结构

| 目录 | 内容 | Git 跟踪 |
|---|---|---|
| `collection/` | 采集与网页/桌面标注工具 | 是 |
| `pipeline/` | 下载、预处理、在线增强与训练缓存代码 | 是 |
| `datasets/` | 下载原包、原始图像、人工标签与数据内的缓存 | 否 |
| `audits/` | 本地数据清理审计记录 | 否 |
| `previews/` | 预览生成脚本；视频等生成产物不上传 | 仅源码与说明 |
| `clarius_sdk/` | 采集需要的厂商 SDK | 按目录现有规则 |

以下命令均从仓库根目录执行。

## 1. 训练入口与标注语义

磁盘目录名与训练参数不完全相同，请按表使用：

| 数据目录 | `--dataset` / `DATASET` 参数 | `label_mode` | 训练 mask id | 标签内容 |
|---|---|---|---|---|
| `data/datasets/Mus-V/` | `musv` | `full3` | 0 / 1 / 2 | 背景、静脉、动脉 |
| `data/datasets/mendeley_data/` | `mendeley` | `artery` | 0 / 2 | 颈总动脉；其余区域不区分背景和静脉 |
| `data/datasets/PMC9883282/` | `pmc9883282` | `full3` | 0 / 1 / 2 | 人工确认的动静脉；原始 PNG 0 / 255 / 128 映射到共享 id |
| `data/datasets/phantom_taobao/` | `phantom_taobao` | `vessel` | 0 / 3 | 未分型仿体血管 |
| `data/datasets/customer_3d_phantom/` | `customer_3d_phantom` | `vessel` | 0 / 3 | 未分型仿体血管 |

模型只输出背景、静脉、动脉三个通道。标签 id 3 表示“已知是血管，未区分类型”，用于 partition CE＋Dice 的边缘概率监督，不是第四个输出通道。实现与数据配置见 [`pipeline/dataPrepare.py`](pipeline/dataPrepare.py)。

ThrombUS、Regional-US、TUS-REC2024 是可选原始视频来源，目前尚未接入这些训练参数；需先完成格式适配与 A/V 人工标注。下载代码不会把它们原有的神经标签、诊断分类或位姿变成分割真值。

PMC 的网页排除标记同时控制图像和 mask 是否用于训练；有效空血管帧保留。修改标签或质量标记后必须重建缓存。复核、恢复与缓存检查见 [网页标注说明](collection/README_WEB_CN.md)。

---

## 2. 从帧到张量

[`pipeline/dataPrepare.py`](pipeline/dataPrepare.py) 把下载好的数据集变成 NPZ 缓存，[`ReadDataset`](pipeline/dataPrepare.py) 再把缓存变成 batch。

### 2.1 先划分，再增强

train/val/test 的划分是在**原始帧**层面完成的，发生在任何增强之前，随机种子固定 `seed=42`。只有 train 划分会被增强；val 和 test 只做 resize 和归一化，别的什么都不做，这样指标才如实反映泛化能力。

```
原始帧（每个数据集）
  ├─ train  70%  → 在线增强，每个 epoch 都是新的
  ├─ val    15%  → 只做 resize + 归一化
  └─ test   15%  → 只做 resize + 归一化
```

Mus-V 是**按扫查序列**划分的，不是按帧。同一次扫查的连续帧几乎是重复的；按帧划分会把同一根血管同时放进 train 和 test。

### 2.2 NPZ 缓存里存的是原图，不是增强副本

这一点最容易搞错。缓存里存的是 **resize 后的原始帧**：

| 字段 | 形状 | dtype | 含义 |
|---|---|---|---|
| `images` | `(N, 576, 544)` | `uint8` | 灰度帧，已完成 resize 和裁剪 |
| `masks` | `(N, 576, 544)` | `uint8` | 共享空间里的类别 id（0/1/2/3） |
| `image_type` | `(N,)` | `str` | 来源标记 |
| `metadata` | `(1,)` | `dict` | 增强配置指纹 |

增强是**在线的，在 `__getitem__` 里做**。`ReadDataset.__len__` 返回 `原图数 × aug_times_train`，同一张原图的每一次抽取都会得到一次独立的随机增强 —— 下个 epoch 又不一样。所以 `aug_times_train` 决定的是「每张原图每个 epoch 产出多少**新鲜**样本」，而不是磁盘上躺着多少份固定拷贝。

训练入口支持 `TRAIN_REPEATS=数据集:次数,...` 覆盖上述默认值；此时长度为 `原图数 × 覆盖次数`。它只改变训练抽样，不修改全局数据配置、NPZ、标签或验证/测试集。调整比例无需再次运行数据导出，也不会生成新的人工标注。

| 数据集 | `aug_times_train` | 原因 |
|---|---|---|
| `phantom_taobao` | 40 | 对冲用，否则仿体在混合训练集里被稀释到约 7% |
| `customer_3d_phantom` | 40 | 同上 |
| `musv` | 8 | 从 3 提到 8 —— 静脉类严重过拟合（train 0.855 / val 0.609） |
| `mendeley` | 5 | 本身就有 1100 帧 |
| `pmc9883282` | 8（历史默认）；本轮通过 `TRAIN_REPEATS` 覆盖为 50 | 120 张训练原图由每轮 960 次提高到 6000 次，保留现有在线增强强度 |

原 v11 训练的历史五域配比（不代表后续人工复核后的实时样本数）：Mus-V 17,624（57.1%）、Mendeley 3,850（12.5%）、淘宝仿体 2,560（8.3%）、自制仿体 840（2.7%）、PMC 6,000（19.4%），合计 **30,874 次/epoch**。两个仿体分别只有 64、21 张训练原图；PMC 有 120 张，但训练部分仅来自受试者20，不能把重复抽样当成新增受试者或独立数据。验证与测试仍各为另外一位受试者的30张标注，用于检查泛化与过拟合。

### 2.3 增强流水线

只作用于 train 样本，顺序如下。几何操作对图像和 mask **同步**生效；外观操作只作用于图像。

| # | 步骤 | 概率 | 图像 | Mask |
|---|---|---|---|---|
| 1 | `isotropic_zoom` —— 单一各向同性缩放因子 | 设了 `aug_scale` 时 | ✅ | ✅ |
| 2 | Resize 到 1.5× 目标尺寸，`INTER_CUBIC` | 必执行 | ✅ | ✅ |
| 3 | 旋转 ±10°，`BORDER_REPLICATE` | 0.5 | ✅ | ✅ |
| 4 | 水平翻转 | 0.5 | ✅ | ✅ |
| 5 | 垂直翻转 | 0.5 | ✅ | ✅ |
| 6 | Resize 到 576×544，`INTER_AREA` | 必执行 | ✅ | ✅ |
| 7 | 亮度 / 对比度 ±10% | 0.6（仿体 0.7） | ✅ | ❌ |
| 8 | 高斯噪声，σ ∈ [0.05, 0.1] | 0.3 | ✅ | ❌ |
| 9 | 随机 gamma | 0.5，设了 `aug_gamma` 时 | ✅ | ❌ |
| 10 | 乘性（散斑）噪声 | 0.4，设了 `aug_speckle` 时 | ✅ | ❌ |
| 11 | 高斯模糊 **或** 锐化 | 设了 `aug_blur` 时 | ✅ | ❌ |
| 12 | 归一化 → 张量 | 必执行 | ✅ | ❌ |

其中两个设计是刻意的，值得保留：

**先放大再缩小。** 第 2 步在旋转**之前**放大到 1.5×，这样旋转时四角有真实像素可取，第 6 步再缩回去。直接在目标尺寸上旋转会留下黑色楔形。

**缩放是手写的，不用 `A.Affine(scale=…)`。** albumentations 会**独立**采样 x 和 y 的缩放比例，这会把一个圆变成 0.70:1 到 1.47:1 之间的任意椭圆。而动脉 vs 静脉是一个**形状**判断 —— 动脉是圆的，静脉会塌陷 —— 各向异性缩放会毁掉模型仅有的那个线索。

第 12 步之后：图像是 `(1, 576, 544)` float32，通过 `(x/255 − 0.5) / 0.5` 落在 `[-1, 1]`；mask 是 `(1, 576, 544)` int64 类别 id，所有外观步骤都不碰它。

### 2.4 缓存失效

`dataPrepare.py` 会在缓存旁边存一份增强配置的指纹。改了增强参数，缓存就失效了：

```bash
python data/pipeline/dataPrepare.py --check-stale       # 只报告，不改动任何东西
python data/pipeline/dataPrepare.py --dataset <name>    # 重建
```

| 参数 | 作用 |
|---|---|
| `--dataset <name>` | 要构建哪个数据集：`phantom_taobao` / `customer_3d_phantom` / `mendeley` / `musv` / `pmc9883282` |
| `--no-png` | 跳过写 PNG 预览（NPZ 照常生成）—— 大数据集上快得多 |
| `--test-loader` | 构建完顺手实例化 DataLoader 并打印 batch 形状 |
| `--check-stale` | 报告哪些缓存已经和配置对不上了 |

### 2.5 添加新数据集

在 [`pipeline/dataPrepare.py`](pipeline/dataPrepare.py) 的 `DATASET_CONFIGS` 里加一条：

```python
"my_dataset": {
    "source_type":     "folder",                       # "csv" | "folder" | "musv"
    "data_dir":        Path("data/datasets/my_dataset"),
    "images_dir":      Path("data/datasets/my_dataset/images"),
    "masks_dir":       Path("data/datasets/my_dataset/masks"),
    "npz_out_dir":     Path("data/datasets/my_dataset"),
    "npz_prefix":      "augmented_my_dataset_1",
    "aug_times_train": 10,                             # 按数据量定
    "mask_class_map":  {0: 0, 255: 3},                 # 磁盘原值 -> 共享类别 id
    "label_mode":      "vessel",                       # full3 | artery | vessel
    "target_size":     (576, 544),
    "color_mode":      "gray",                         # "gray" | "rgb_to_gray"
    "crop":            None,                           # None | (上, 下, 左, 右)
    "seed":            42,
},
```

`label_mode` 是最关键的一项 —— 它声明这个数据集的 mask 实际能确定类别划分中的哪些组，也就决定了 loss 被允许监督什么。详见[顶层 README](../README.md) 的 loss 章节。

然后构建：

```bash
python data/pipeline/dataPrepare.py --dataset my_dataset --no-png
```

---

## 3. 采集

[`collection/`](collection/) 是采集侧，和训练无关 —— 只有要采新数据时才需要它。

| 文件 | 作用 |
|---|---|
| [`collect_clarius.py`](collection/collect_clarius.py) | 通过 Cast API 从 Clarius 探头采集 |
| [`collect_a325.py`](collection/collect_a325.py) | 从 A325 采集卡采集 |
| [`label.py`](collection/label.py) | 圆刷标注工具（PySide6） |
| [`config.py`](collection/config.py) | 采集侧路径，以及 `label.py` 画上去的 mask 取值 |

`config.py` 和 `pipeline/dataPrepare.py` 必须在 mask 取值上保持一致：`label.py` 画上去的值如果没有任何 `mask_class_map` 提到它，训练时会被静默丢弃。

---

## 4. Clarius SDK

[`clarius_sdk/`](clarius_sdk/) 存放 Clarius Cast 原生 API —— 动态库、C 头文件、上游 `cast` submodule，以及 Clarius 官方的示例程序。它**仅**用于原生 API 开发和采集；训练和推理完全不碰它。详见 [`clarius_sdk/README.md`](clarius_sdk/README.md)。
