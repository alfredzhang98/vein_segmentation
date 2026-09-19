# 血管分割 (Vein Segmentation)

> English: [README.md](README.md)

面向穿刺引导的超声血管分割，围绕 Clarius HD3 L7 探头构建。本仓库覆盖从采集到可部署模型的完整链路：采集、标注、数据集准备、训练、评估、出图，以及独立推理。

网络是一个 U-Net，输出**三个通道 —— 背景、静脉、动脉**，并在四个**对「mask 意味着什么」意见不一致**的数据集上联合训练。**解决这个不一致，而不是网络结构本身，才是本仓库的主题。**

---

## 问题，以及解决它的目标函数

四个训练集标注的根本不是同一件事：

| 数据集 | mask 标了什么 | 留下什么未定 |
|---|---|---|
| Mus-V | 静脉和动脉，分别标注 | 无 |
| `phantom_taobao`、`customer_3d_phantom` | 一根凝胶管 | 这根管子算静脉还是动脉 |
| Mendeley CCA | 颈总动脉 | 颈内静脉 —— 通常就在画面里，却**没有标注** |

把它们汇总起来用一个普通的 loss 训练，会把 CCA 里未标注的颈内静脉当作**背景**来监督，于是教会模型「静脉长得像背景」—— 静脉类被本该帮助它的数据摧毁。

取而代之的做法是：把每份标注视为**只能把三个类别分辨到某个划分（partition）`𝒢` 的精度**。记 `q_g = Σ_{c∈g} p_c` 为组 `g ∈ 𝒢` 的边缘概率，`y_g ∈ {0,1}` 为该组的标签，复合的交叉熵–Dice 目标函数作用在这些边缘概率上：

```
L = − (1/|Ω|) Σ_{i∈Ω} Σ_{g∈𝒢} y_{g,i} log q_{g,i}
    + (1/|𝒢⁺|) Σ_{g∈𝒢⁺} ( 1 − (2 Σ_i q_{g,i} y_{g,i} + ε) / (Σ_i q_{g,i} + Σ_i y_{g,i} + ε) )
```

其中 `Ω` 是一张图的像素集合，`ε = 1e-6`，`𝒢⁺` 是 `𝒢` 去掉含背景类的那一组。

| 数据集 | `label_mode` | 划分 `𝒢` | 被监督的边缘概率 |
|---|---|---|---|
| Mus-V | `full3` | `{bg}, {vein}, {artery}` | 三个都监督（普通 3 类 CE + 静脉/动脉上的 Dice） |
| 仿体 | `vessel` | `{bg}, {vein, artery}` | `q = p₁ + p₂` |
| Mendeley CCA | `artery` | `{bg, vein}, {artery}` | `q = p₂` |

由此直接得到两个推论，而这两个推论正是全部要点：

- 对仿体，loss **只**依赖 `p₁ + p₂`，因此这个和如何在静脉与动脉之间拆分是不受约束的 —— 拆分方式由从 Mus-V 学到的先验决定。这正是你能对着一根仿体管子问「它看起来更像静脉还是动脉」的原因。
- 对 CCA，loss **只**依赖 `p₂`，因此它对剩余概率如何在背景与静脉之间分配是不变的。在这里预测出一根未标注的颈内静脉，代价**恰好为零**。

当 `𝒢` 只有两组时，第一项严格就是二元交叉熵，第二项退化为单个 Dice 项，于是整个表达式还原成常规的复合 BCE–Dice loss。实现在 [`models/unet/train.py`](models/unet/train.py) 的 `partial_label_loss()`；全部用 `log_softmax` 和 `logsumexp` 计算，所以 `log(p₁ + p₂)` 是精确的，在 AMP 下也安全。

---

## 项目结构

```
vein_segmentation/
├── .env                        # 全部训练超参 —— 改这里，不要改 train.py
│
├── data/                       # 一切和数据有关 —— 见 data/README.md
│   ├── collection/             #   采集（Clarius Cast、A325）+ 圆刷标注工具
│   ├── pipeline/               #   预处理、增强、NPZ 缓存、ReadDataset
│   ├── datasets/               #   数据本体 —— 不跟踪，需自行下载
│   ├── clarius_sdk/            #   Clarius 原生 API —— 仅采集需要，训练用不到
│   └── samples/                #   几张示例帧，让下面的命令开箱即跑
│
├── models/                     # 一个架构一个自包含文件夹
│   └── unet/                   #   U-Net（源自 milesial/Pytorch-UNet）
│       ├── model.py  parts.py  #     网络与其构件
│       ├── train.py            #     部分标签 loss、验证、checkpoint
│       ├── test.py             #     留出集评估、后处理、FPS
│       └── infer.py            #     独立推理 + 几何测量（无训练侧依赖）
│
├── analysis/                   # 测量，不是训练
│   └── unet_analysis/          #   接收 U-Net checkpoint —— 见下方说明
│       ├── bench_fps.py        #     端到端延迟 / FPS
│       ├── summarize_model.py  #     参数与配置汇总表
│       ├── predict_geometry.py #     mask → 质心、横向偏移、深度、半径
│       └── probe_generalization.py   # checkpoint 的跨域探针
│
├── results/                    # 值得留存的产出 —— 跟踪，一个架构一个子目录
│   └── unet/                   #   U-Net 结果
│       ├── README.md           #     结果、实验记录与论文素材
│       ├── README_CN.md        #     中文版
│       ├── figure_style.py  plot_results.py  plot_segmentation_samples.py
│       └── figures/            #     生成的图，含已发表的那张
│
├── gpu_utils.py                # 空闲 GPU 选择，全仓库共用
└── wandb/                      # W&B 运行数据 —— 不跟踪
```

**所有脚本都从仓库根目录运行。** `dataPrepare.py` 里的数据集路径是相对路径（`data/datasets/...`）；每个脚本都会自己把仓库根加进 `sys.path`，所以无论用哪种方式启动，`data.pipeline.*` 和 `models.<arch>.*` 都能一致解析。

**新增一个架构**的做法是：加一个 `models/<arch>/`，里面放它自己的 `model.py`、`train.py`、`test.py` 和 `infer.py`，然后把 [`.env`](.env) 里的 `CHECKPOINT_DIR` 指向 `models/<arch>/checkpoints`。`data/` 下面一行都不用改 —— 数据管线、标签空间和数据集配置是共用的。`analysis/` 遵循同样的约定：今天是 `analysis/unet_analysis/`，以后在旁边加 `analysis/<arch>_analysis/`。

**`analysis/` 目前是架构相关的。** 四个脚本都直接构造 `UNet`，而且 `summarize_model.py` 打印的是手写的 U-Net 拓扑描述。它们放在顶层，是因为「测量延迟、几何和跨域泛化」这件事本身是通用需求 —— 但代码还不通用。

---

## 安装

```bash
git clone --recursive https://github.com/alfredzhang98/vein_segmentation.git
cd vein_segmentation
pip install -r requirements.txt
```

---

## 数据

`data/datasets/` 下的内容一律不跟踪 —— 四个数据集里有两个的许可不允许在此转载。逐个数据集的下载链接和它们各自必须落到的确切目录结构，见 [data/README_CN.md](data/README_CN.md)。

| 目录 | 来源 | `label_mode` | 训练样本/epoch | Val |
|---|---|---|---|---|
| `data/datasets/Mus-V/` | 公开，Kaggle/Springer | `full3` | 17 624 (70.9%) | 911 |
| `data/datasets/mendeley_data/` | 公开，Mendeley Data | `artery` | 3 850 (15.5%) | — |
| `data/datasets/phantom_taobao/` | 自采，商用仿体 | `vessel` | 2 560 (10.3%) | 13 |
| `data/datasets/customer_3d_phantom/` | 自采，自制明胶/琼脂仿体 | `vessel` | 840 (3.4%) | 4 |

两个仿体的验证划分被**合并**成一个 17 图指标 —— 4 图的划分单独打分只是噪声。

数据就位后，构建增强 NPZ 缓存：

```bash
python data/pipeline/dataPrepare.py --dataset musv
python data/pipeline/dataPrepare.py --dataset mendeley --no-png
python data/pipeline/dataPrepare.py --dataset phantom_taobao
python data/pipeline/dataPrepare.py --dataset customer_3d_phantom
python data/pipeline/dataPrepare.py --check-stale     # 哪些缓存已和配置对不上
```

---

## 快速开始

全部超参、数据集选择和 checkpoint 判据都在 [`.env`](.env) 里 —— 随仓库发布的那组值就是产出已发布 checkpoint 的那组。不想要实验记录就设 `WANDB_ENABLE=false`。

```bash
# 训练（读 .env；自己挑一张空闲 GPU）
python models/unet/train.py

# 在留出 test 划分上评估一个 checkpoint
python models/unet/test.py --ckpt models/unet/checkpoints/unet_v10.pth \
                           --dataset musv mendeley phantom_taobao+customer_3d_phantom

# 延迟 / 吞吐
python analysis/unet_analysis/bench_fps.py --ckpt models/unet/checkpoints/unet_v10.pth

# 从单帧算几何量（质心、横向偏移、深度、半径）
python analysis/unet_analysis/predict_geometry.py --ckpt models/unet/checkpoints/unet_v10.pth \
                                    --image data/samples/0016.png \
                                    --dataset phantom_taobao

# 重新生成已发表的图
python results/unet/plot_segmentation_samples.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --n 2 --seed-phantom 41 --seed-musv 13 --seed-cca 116
```

独立推理，不带任何训练侧 import：

```python
from models.unet.infer import UNetInferencer

inf  = UNetInferencer("models/unet/checkpoints/unet_v10.pth")
mask = inf.predict("frame.png")
g    = inf.measure(mask, mm_per_px=0.0832)          # 4 cm 深度；5 cm 用 0.104
print(g["artery"]["lateral_mm"], g["artery"]["depth_mm"], g["artery"]["radius_mm"])
```

---

## 结果

参考 checkpoint：**`models/unet/checkpoints/unet_v10.pth`**（U-Net，`base_ch=32`，7 762 531 参数，dropout 0.3，epoch 19，best PRIMARY 0.8622）。Test 划分在序列级和受试者级上都是不相交的。

| Test 集 | 类别 | Dice |
|---|---|---|
| `customer_3d_phantom` —— 引导实验实际运行的域 | vessel | **0.970** |
| `phantom_taobao` | vessel | **0.866** |
| Mendeley CCA | artery | **0.956** |
| Mus-V | artery | **0.882** |
| Mus-V | vein | **0.617** |

Dice 是在部署流水线所做的形态学后处理**之后**测的，因为那才是机器人真正消费的东西；`models/unet/test.py` 会同时打印原始 argmax 的对照表，这样后处理就藏不住模型的真实行为。复现：

```bash
python models/unet/test.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --dataset musv mendeley customer_3d_phantom phantom_taobao --min-area 150
```

`--min-area 150` 不能省：这个参数**默认是 0**，也就是完全关闭后处理，而上表给的是部署流水线实际产出的后处理结果。不加它静脉会是 0.612 而不是 0.617。

### 延迟

在空闲的 NVIDIA H200 NVL 上实测，batch size 1，开 AMP，预热 30 帧后计时 200 帧，用真实图像而非噪声：

| 阶段 | p50 | p95 | 200 帧最差 | FPS |
|---|---|---|---|---|
| 网络前向 | 2.60 ms | 2.61 ms | 2.62 ms | 385 |
| + `fit` / `unfit`（`predict`） | 2.89 ms | 2.90 ms | 3.26 ms | 345 |
| **+ 后处理 + 几何（端到端）** | **5.57 ms** | **5.65 ms** | **6.20 ms** | **179** |

端到端是**每帧 5.6 ms，即 179 fps** —— 远高于超声帧率，所以分割不是引导回路的瓶颈。注意时间花在哪：后处理和矩分析占 **2.7 ms，是总预算的 48%**，而且它们跑在 CPU 上的 OpenCV 里，换更好的 GPU 不会变快。在你自己的硬件上测：

```bash
python analysis/unet_analysis/bench_fps.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --frame data/samples/0016.png --n 200
```

![segmentation samples](results/unet/figures/segmentation_samples.png)

三行依次是 Mus-V、Mendeley CCA 和仿体。注意 CCA 那几行：模型在颈动脉旁边勾出了一根静脉（蓝色），而真值**没有**标它。这正是部分标签目标函数按设计工作的表现 —— CCA 的 loss 对静脉是不变的，所以在那里预测静脉不花任何代价，而静脉先验是从 Mus-V 迁移过来的。

全分辨率的已发表图保存在 [`results/unet/figures/segmentation_samples.svg`](results/unet/figures/segmentation_samples.svg)。

**关于静脉分数。** 0.617 是被*检出*而非*勾勒*所支配的：在**检出**了静脉的那 81.1% 的帧上，Dice 是 0.751，而同一判据下动脉是 0.905。静脉薄壁、在探头压力下会塌陷 —— 在最扁的那 20% 截面上召回率降到 0.46，且有 14 帧静脉被压至零截面，这是动脉从不出现的模式。Mus-V 的 105 次扫查只来自 11 名志愿者，这限制了单帧模型能学到多少这类形变。完整分析见 [results/unet/README_CN.md](results/unet/README_CN.md)。

---

## 部署

### 后处理

部署路径和评估路径共用同一套形态学：开运算（3×3 椭圆核）→ 闭运算（5×5）→ 每类只保留最大连通域 → 丢弃小于 `min_area` 的区域。[`models/unet/infer.py`](models/unet/infer.py) 的 `clean()` 到此为止（`min_area=300`）；[`models/unet/test.py`](models/unet/test.py) 的 `clean_binary()` 在此之上还加了锚点和距离门控（`min_area=150`，`max_dist=40`）。上面的 Dice 数字是后处理**之后**测的，因为那是机器人消费的东西，而 `test.py` 会并排打印原始 argmax 表，这样后处理藏不住模型的行为。

### 无血管帧上的假阳性

Dice 只回答「有血管的时候勾得准不准」。部署引入了训练从不打分的另一个问题：探头压在没有血管的组织上时，模型会不会仍然亮？两个抓手，按有用程度排序：

1. **最小连通域面积。** 血管远大于一个散斑点，而 `clean()` 已经在过滤小于 `min_area` 的区域。把它调高到无血管帧上的假阳性消失，然后确认真实帧上的 Dice 没有动。
2. **置信度裕度。** 模型在三个 softmax 通道上取普通 argmax；要求胜出类别超过一定裕度才被接受，可以用召回率换精确率。

第二次仿体采集里的 40 张无血管帧（`meta_phantom_taobao_1.csv` 里的 `mask_status=test`，对照 92 张标注为 `true` 的帧）就是为这个测量准备的。它们不在任何训练或测试划分里。

> **不要在调参集上报告结果。** 一旦你在这些帧上调了 `min_area` 或裕度，它们就变成了验证集。最终数字必须来自 test 划分，假阳性率单独另报。

### 阈值敏感性

在更早的二分类模型上测过：τ ∈ {0.25, 0.50, 0.75} 范围内，仿体和 CCA 两个 test 集上的 F1 变化都小于 0.005 —— 预测概率集中在 0 和 1 附近，所以工作点从来不敏感。已发布的模型是三分类、用 argmax，根本没有阈值要选；这个测量记录在此，是因为它正是那个问题被关闭的理由。

---

## 实时演示

https://github.com/user-attachments/assets/f3ac8a1e-8759-4e72-b5cd-6a491dfb8b4b

---

## 硬件

- **超声**：Clarius HD3 L7
- **采集卡**：A325（Clarius Cast API 的替代方案）
- **标定**：72.35 µm/px（Clarius 原生）/ 83.78 µm/px（A325，裁剪后）

网络是纯粹的像素到像素映射。物理尺度只在预测 mask 被还原回原生图像网格**之后**才介入，所以单个标定常数就能精确换算两个轴。

---

## 许可

以 [CC BY-NC 4.0 许可](LICENSE)（署名–非商业性使用）发布。

| 第三方数据集 | 许可 |
|---|---|
| Common Carotid Artery Ultrasound Images (Mendeley) | CC BY 4.0 |
| Carotid and Femoral Vessel Ultrasound Dataset (Mus-V) | CC BY-NC 4.0 —— 仅限非商业用途 |

---

## 参考

- **U-Net 架构** — [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- **Common Carotid Artery Ultrasound Images** — [Mendeley Data](https://data.mendeley.com/datasets/d4xt63mgjm/1)
- **Carotid and Femoral Vessel Ultrasound Dataset (Mus-V)** — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_61) · [Kaggle](https://www.kaggle.com/datasets/fa8b3e1386722702d9c80a7d2d10d5d50eef20d14a604078b38d01c66fd9f356)
- **Albumentations** — [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
- **Weights & Biases** — [wandb.ai](https://wandb.ai)
