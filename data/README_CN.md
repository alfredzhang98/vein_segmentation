# 数据

所有和数据相关的东西都在这个目录下：怎么采集、怎么变成训练张量、数据集放在哪，以及采集脚本依赖的厂商 SDK。

> English: [README.md](README.md)

## 目录结构

| 目录 | 内容 | git 跟踪 |
|---|---|---|
| [`collection/`](collection/) | 采集与标注 —— Clarius Cast、A325 采集卡、圆刷标注工具 | 是 |
| [`pipeline/`](pipeline/) | 预处理、数据增强、NPZ 缓存、`Dataset` 类 | 是 |
| [`datasets/`](datasets/) | 实际数据。需自行下载，见 §1 | **否** |
| [`clarius_sdk/`](clarius_sdk/) | Clarius Cast 原生 API。**仅**采集时需要，训练推理用不到 | 是 |
| [`samples/`](samples/) | 几张示例帧，让 README 里的演示命令开箱即跑 | 是 |

所有命令都从**仓库根目录**执行。`pipeline/dataPrepare.py` 里的路径是相对路径（`data/datasets/...`）。

---

## 1. 数据集

`datasets/` 下的内容一律不跟踪。四个数据集里有两个的许可不允许在此转载，另外两个体积太大。请自行下载并放到下面给出的**确切路径**。

目录名有意义：它们是 `DATASET_CONFIGS`（[`pipeline/dataPrepare.py`](pipeline/dataPrepare.py)）的 key，也是 [`.env`](../.env) 里 `DATASET` / `VAL_DATASET` 接受的取值。

| 目录 | 来源 | `label_mode` | Mask id | 标注了什么 |
|---|---|---|---|---|
| `Mus-V/` | Kaggle（公开） | `full3` | 0 / 1 / 2 | 背景、静脉、动脉 —— 三类齐全 |
| `mendeley_data/` | Mendeley Data（公开） | `artery` | 0 / 2 | 只标了颈总动脉 |
| `phantom_taobao/` | 自采 | `vessel` | 0 / 3 | 凝胶管，类型未定 |
| `customer_3d_phantom/` | 自采 | `vessel` | 0 / 3 | 凝胶管，类型未定 |

Mask id `3` 表示「是血管，但没标类型」，它**永远不会**作为模型的输出通道出现 —— 模型只输出 3 个通道（背景 / 静脉 / 动脉）。从磁盘上的原始值到这套共享 id 的映射是 `mask_class_map`，在 `DATASET_CONFIGS` 里逐数据集声明。

### 1.1 `phantom_taobao/` 和 `customer_3d_phantom/` —— 自采

从 [Google Drive](https://drive.google.com/drive/folders/1LPwezlAUmxWnUo791RM3OPAhGQHq5r0v?usp=sharing) 下载，**直接解压到 `datasets/` 里** —— 不要弄出嵌套的 `datasets/data/...`。

> **解压后要改名。** 压缩包用的是旧名字。`phantom_1` 是商用仿体，必须改成 **`phantom_taobao`**；`phantom_2` 是自制仿体，必须改成 **`customer_3d_phantom`**。用旧名字代码找不到。

```
data/datasets/
├── phantom_taobao/                      # 商用仿体，Clarius Cast 采集
│   ├── images/                          #   原始灰度帧
│   ├── masks/                           #   标注的二值 mask
│   ├── images_aug/  masks_aug/          #   dataPrepare.py 可选输出的 PNG 预览
│   └── meta_phantom_taobao_1.csv        #   逐图元数据（micropixel、深度、增益）
│
└── customer_3d_phantom/                 # 自制明胶+琼脂+盐水仿体，A325 采集
    ├── images/  masks/
    ├── images_aug/  masks_aug/
    └── meta_customer_3d_phantom_1.csv
```

```bash
python data/pipeline/dataPrepare.py --dataset phantom_taobao
python data/pipeline/dataPrepare.py --dataset customer_3d_phantom
```

- `phantom_taobao` —— 商用仿体，[淘宝链接](https://item.taobao.com/item.htm?id=762322402710)
- `customer_3d_phantom` —— 自制（明胶 + 琼脂 + 盐水），3D 打印血管芯

### 1.2 `mendeley_data/` —— 颈总动脉超声图像

**未包含。** 从 [Mendeley Data](https://data.mendeley.com/datasets/d4xt63mgjm/1)（CC BY 4.0）下载，解压后路径必须正好是：

```
data/datasets/mendeley_data/
└── Common Carotid Artery Ultrasound Images/
    ├── US images/              # 1100 帧（PNG, 709x749x3）
    └── Expert mask images/     # 1100 张专家 mask，文件名一一对应
```

```bash
python data/pipeline/dataPrepare.py --dataset mendeley --no-png
```

**规格** —— 1100 图 + 1100 专家 mask，709×749×3；迈瑞 UMT-500Plus 配 L13-3s 线阵探头；11 名受试者；CC BY 4.0。

**为什么 `label_mode = artery`。** 专家 mask 只标了颈总动脉，别的什么都没标。颈内静脉经常就在画面里，却没有标注，被归进了「背景」区域。把这些像素当背景来监督，等于在教模型「静脉长得像背景」。所以这个目标函数只约束动脉边缘概率 `p₂`，对剩下的概率在背景和静脉之间怎么分完全不敏感 —— 在这里预测出一根没标注的颈内静脉，代价恰好是零。

> loss 分三处记录：**数学形式**在[顶层 README](../README_CN.md)；**每个数据集对应哪个划分**就是上表的 `label_mode` 列；**实现**在 [`models/unet/README_CN.md`](../models/unet/README_CN.md#loss--partial_label_loss)。

### 1.3 `Mus-V/` —— 颈动脉与股血管超声数据集

**未包含。** 从 [Kaggle](https://www.kaggle.com/datasets/fa8b3e1386722702d9c80a7d2d10d5d50eef20d14a604078b38d01c66fd9f356)（CC BY-NC 4.0，**仅限非商用**）下载，解压后路径为：

```
data/datasets/Mus-V/
└── Multimodal Ultrasound Vascular Image Segmentation/
    └── ...                     # 保持原样
```

```bash
python data/pipeline/dataPrepare.py --dataset musv
```

**规格** —— 3114 帧（2203 训练 + 911 验证），来自 105 次探头扫查、11 名志愿者；Angell Pioneer H20；CC BY-NC 4.0。

**类别身份的确认。** Mask 本身已经是 `{0, 1, 2}` 标签图，所以 `mask_class_map` 是恒等映射。类别 1 是**静脉**、类别 2 是**动脉**，最初是从面积统计推断出来的 —— 类别 1 在 272 帧里面积塌缩到零，且面积方差是类别 2 的 3.6 倍，也就是说它是可压缩的那个 —— 随后由 Mendeley 独立验证：从 Mus-V 学到的类别 2 通道，在 Mendeley 专家**动脉** mask 上达到 Dice 0.956。如果两者搞反了，这个数字会接近零。

这是唯一一个把静脉和动脉分开标注的数据集，因此也是仿体和 CCA 数据集所继承的「静脉/动脉先验」的唯一来源。

### 1.4 `PMC9883282/` —— 颈内静脉压力塌缩记录（可选，**无标注**）

不用于训练。来自 Scientific Reports 一项关于探头压力下颈内静脉塌缩的研究的补充数据：6 个 MATLAB v7.3 文件，每个含 `ultrasound_images (N, 800, 600) uint8`、`force_data` 和 `time_data`，覆盖该研究 27 名受试者中的 3 名。

**它完全没有任何 mask。** 它进不了训练目标 —— 一个数据集至少要能确定划分中的一个组，才能监督任何东西，而这个数据集一个都确定不了。保留它仅仅是作为 [`results/unet/README.md`](../results/unet/README.md) 里讨论的「静脉压缩失败模式」的参考。真要用还需要裁剪（原帧是带 UI 叠层的整屏截图）和重采样（力是 ~50 Hz，视频是 29 Hz）。

```bash
curl -O https://pmc-oa-opendata.s3.amazonaws.com/PMC9883282.1/41598_2022_22867_MOESM1_ESM.zip
```

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

| 数据集 | `aug_times_train` | 原因 |
|---|---|---|
| `phantom_taobao` | 40 | 对冲用，否则仿体在混合训练集里被稀释到约 7% |
| `customer_3d_phantom` | 40 | 同上 |
| `musv` | 8 | 从 3 提到 8 —— 静脉类严重过拟合（train 0.855 / val 0.609） |
| `mendeley` | 5 | 本身就有 1100 帧 |

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
| `--dataset <name>` | 要构建哪个数据集：`phantom_taobao` / `customer_3d_phantom` / `mendeley` / `musv` |
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
