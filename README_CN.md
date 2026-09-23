# 血管分割与时序预测

当前单帧基线为 **U-Net v11**；时序研究模型为 **双尺度 ConvGRU-U-Net**。v11 重训候选有跨域退化，未覆盖原权重；时序候选尚未通过验收。

| 内容 | 代码与训练入口 | 结果入口 |
|---|---|---|
| 单帧分割 | [models/unet](models/unet/README_CN.md) | [v11 重训结果](results/unet/V11_REFRESH_CN.md) · [Mus-V 视频](results/unet/generated/musv_review/index.html) |
| 时序跟踪与预测 | [models/temporal](models/temporal/README_CN.md) | [时序结果](results/temporal/RESULTS_CN.md) · [完整视频](results/temporal/generated/video/index.html) |
| 数据准备与标注 | [data](data/README_CN.md) | 本地清理审计：`data/audits/` |

[目录与 Git 规则](docs/DIRECTORY_LAYOUT_CN.md) · [研究思路](docs/ultrasound_world_model.html) · [English](README.md)

**首次 clone：[数据集下载与准备指南](data/VIDEO_DATASETS_CN.md)**，包含 Mus-V 下载链接、其他公共来源的下载命令、路径与空间要求。

PMC 本地队列扩充至 362 张，人工复核后的有效数量以 metadata CSV 为准。官方原始包不含本项目人工标签，clone 也不会带上这份标注快照；排除无效帧时保留原图与 mask，不把有效空血管帧当作无效图。

## 问题，以及解决它的目标函数

当前五个数据集的标注语义如下：

| 数据集 | mask 标了什么 | 留下什么未定 |
|---|---|---|
| Mus-V | 静脉和动脉，分别标注 | 无 |
| `PMC9883282` | 318 张有效、人工确认的静脉和动脉 mask | 其余有效帧未标注，不能当作空 mask；无接触帧排除 |
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
| Mus-V、PMC 已确认标注 | `full3` | `{bg}, {vein}, {artery}` | 三个都监督（普通 3 类 CE + 静脉/动脉上的 Dice） |
| 仿体 | `vessel` | `{bg}, {vein, artery}` | `q = p₁ + p₂` |
| Mendeley CCA | `artery` | `{bg, vein}, {artery}` | `q = p₂` |

由此直接得到两个推论，而这两个推论正是全部要点：

- 对仿体，loss **只**依赖 `p₁ + p₂`，因此这个和如何在静脉与动脉之间拆分是不受约束的；动静脉先验来自完整标注域（v10 为 Mus-V，下一轮加入 PMC）。
- 对 CCA，loss **只**依赖 `p₂`，因此它对剩余概率如何在背景与静脉之间分配是不变的。在这里预测出一根未标注的颈内静脉，代价**恰好为零**。

当 `𝒢` 只有两组时，第一项严格就是二元交叉熵，第二项退化为单个 Dice 项，于是整个表达式还原成常规的复合 BCE–Dice loss。实现在 [`models/unet/train.py`](models/unet/train.py) 的 `partial_label_loss()`；全部用 `log_softmax` 和 `logsumexp` 计算，所以 `log(p₁ + p₂)` 是精确的，在 AMP 下也安全。

---

## 项目结构

```text
vein_segmentation/
├── configs/                  # 训练配置；temporal.json / v11_refresh.env
├── data/
│   ├── collection/           # 采集与标注工具
│   ├── pipeline/             # 数据预处理代码，保留此目录
│   ├── datasets/             # 原图与标签，本地、不上传
│   ├── cache/                # 可重建缓存，本地、不上传
│   └── audits/               # 数据清理明细，本地、不上传
├── models/
│   ├── unet/                 # 单帧模型：model/train/evaluate/infer
│   ├── temporal/             # 当前 ConvGRU 时序模型：同样四个入口
│   └── losses.py             # 共享 partition loss
├── results/
│   ├── unet/                 # 单帧报告与生成代码
│   │   ├── runs/             # 本地运行记录，不上传
│   │   └── generated/        # 本地页面、视频、评估，不上传
│   └── temporal/             # 时序报告与生成代码；相同子目录约定
├── tests/                    # 本地测试，不上传
└── docs/                     # 研究思路与目录约定
```

从本项目根目录运行命令。`models/` 放模型实现和本地 checkpoint；`results/` 放结果摘要、出图代码及本地产物。Stage1 是实验阶段名称，不再另设 `models/stage1/`；跨模型结果不再塞进日期命名的 pipeline 目录。

单元测试、所有 `runs/`、生成页面、权重与缓存由 `.gitignore` 排除。正式评估程序命名为 `evaluate.py`，属于应保留的源码。

---

## 安装

```bash
git clone --recursive https://github.com/alfredzhang98/vein_segmentation.git
cd vein_segmentation
pip install -r requirements.txt
```

---

## 数据

`data/datasets/` 下的内容一律不跟踪；请按各自来源和许可获取数据。五个数据集的下载说明与目录结构见 [data/README_CN.md](data/README_CN.md)。下表的旧四域采样量和验证数量为 v10 历史配置，不代表下一轮五域训练的配比。

| 目录 | 来源 | `label_mode` | v10 训练样本/epoch | Val |
|---|---|---|---|---|
| `data/datasets/Mus-V/` | 公开，Kaggle/Springer | `full3` | 17 624 (70.9%) | 911 |
| `data/datasets/mendeley_data/` | 公开，Mendeley Data | `artery` | 3 850 (15.5%) | — |
| `data/datasets/phantom_taobao/` | 自采，商用仿体 | `vessel` | 2 560 (10.3%) | 13 |
| `data/datasets/customer_3d_phantom/` | 自采，自制明胶/琼脂仿体 | `vessel` | 840 (3.4%) | 4 |
| `data/datasets/PMC9883282/` | 公开原始序列 + 本地人工标注 | `full3` | 未参与 v10 | 30 张人工标签 |

v10 配置中两个仿体的验证划分合并成一个 17 图指标。新的实验须单独记录有效配置和各域结果。

### PMC9883282：原始序列与人工标注

数据来自[颈内静脉接触力—塌缩研究的补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883282/)，包含 3 位受试者、6 个 MAT、5215 帧左颈部超声，以及独立记录的 force/time。原始包没有分割标签；本地已完成 **180 张人工 A/V 标注**，按受试者划分为 **train 120 / val 30 / test 30**。

其中 151 张包含动静脉、6 张仅有动脉、23 张为空。这些缺失或空 mask 是有效标注；未标注帧则不能当作背景。PMC 沿用现有 `full3` 混合损失，不增加“血管”输出类别。已删除的 `suggestions/` 只是预标注草稿，训练使用已确认的 `masks/`。

当前人工标签和 PMC 训练缓存均已准备好（train 120 / val 30 / test 30），五域缓存配置指纹检查通过；v11 已训练并完成五域对照评估。图像逐帧时间戳和 pose 未提供，force 与图像的同步尚未确认，因此暂不作为逐帧力监督。

- [原始数据检查、下载与视频预览生成](data/previews/PMC9883282/README_CN.md)
- [Web 标注与人工 mask 复查](data/collection/README_WEB_CN.md)
- [五域训练下一步与完整研究路线](docs/ultrasound_world_model.html#next-steps)

数据就位后，构建 NPZ 缓存（训练增强在线执行）：

```bash
python data/pipeline/dataPrepare.py --dataset musv
python data/pipeline/dataPrepare.py --dataset mendeley --no-png
python data/pipeline/dataPrepare.py --dataset phantom_taobao
python data/pipeline/dataPrepare.py --dataset customer_3d_phantom
python data/pipeline/dataPrepare.py --dataset pmc9883282 --validate-only
python data/pipeline/dataPrepare.py --dataset pmc9883282 --no-png
python data/pipeline/dataPrepare.py --check-stale     # 哪些缓存已和配置对不上
```

---

## 快速开始

全部超参、数据集选择和 checkpoint 判据都在 [`.env`](.env) 里 —— 随仓库发布的那组值就是产出已发布 checkpoint 的那组。不想要实验记录就设 `WANDB_ENABLE=false`。

默认 `.env` 和通用模板 [`configs/train.env`](configs/train.env) 仍选择旧四域，**不会自动加入 PMC**。五域实验的数据组合、验证权重及从头训练／微调设置见[主文档执行流程](docs/ultrasound_world_model.html#next-steps)；下方默认训练命令沿用四域选择。

五域新实验采用 `TRAIN_REPEATS=pmc9883282:50`：PMC 的训练抽样占比由 3.7% 提高到 **19.4%**（120 张原图×50＝6000 次/epoch），其余四域抽样次数不变；无需重建缓存。本轮已从头训练并在第 62 轮早停，保留第 32 轮最佳模型。权重统一存放为 `models/unet/checkpoints/unet_v10.pth` 与 `unet_v11.pth`；旧中断实验及两个版本子目录已清理。抽样比例与 `VAL_WEIGHTS` 独立，验证权重本轮保持不变。完整可运行 Bash 见主文档。

```bash
# 训练（读 .env；自己挑一张空闲 GPU）
python models/unet/train.py

# 在留出 test 划分上评估一个 checkpoint
python models/unet/evaluate.py --ckpt models/unet/checkpoints/unet_v10.pth \
                           --dataset musv mendeley phantom_taobao+customer_3d_phantom

# 延迟 / 吞吐
python analysis/unet_analysis/bench_fps.py --ckpt models/unet/checkpoints/unet_v10.pth

# 从单帧算几何量（质心、横向偏移、深度、半径）
python analysis/unet_analysis/predict_geometry.py --ckpt models/unet/checkpoints/unet_v10.pth \
                                    --image data/samples/0016.png \
                                    --dataset phantom_taobao

# 生成v11八行展示图（只展示一个phantom，保留历史旧图）
python results/unet/plot_segmentation_samples.py --ckpt models/unet/checkpoints/unet_v11.pth \
       --datasets musv mendeley pmc9883282 customer_3d_phantom \
       --n 2 --showcase-datasets musv mendeley pmc9883282 --showcase-min-dice 0.90 \
       --out results/unet/figures/segmentation_samples_unet_v11_showcase
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

上表为历史 v10 实验数字，不代表当前后处理版本。新的 raw / 旧清理 / 联合修复对照见[后处理更新](results/unet/BASELINE_COMPARISON_CN.md)。使用当前规则重新评估：

```bash
python models/unet/evaluate.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --dataset musv mendeley customer_3d_phantom phantom_taobao --min-area 4
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

上图为保留的历史结果。本轮仅保留[八行v11展示图](results/unet/figures/segmentation_samples_unet_v11_showcase.png)及其选帧JSON，依次展示Mus-V、Mendeley、PMC和Custom 3D phantom，使用最终联合后处理。Mus-V、Mendeley和PMC为每类Dice均不低于0.90且兼顾画面差异的成功样例，phantom固定seed；展示分数不代表数据集均值，全测试集统计仍包含两个phantom。最终展示PNG及选帧JSON已从Git忽略规则中放行，可随仓库提交，可用上方命令生成；需先准备数据、缓存和权重，已有输出不会覆盖。

三行依次是 Mus-V、Mendeley CCA 和仿体。注意 CCA 那几行：模型在颈动脉旁边勾出了一根静脉（蓝色），而真值**没有**标它。这正是部分标签目标函数按设计工作的表现 —— CCA 的 loss 对静脉是不变的，所以在那里预测静脉不花任何代价，而静脉先验是从 Mus-V 迁移过来的。

全分辨率的已发表图保存在 [`results/unet/figures/segmentation_samples.svg`](results/unet/figures/segmentation_samples.svg)。

**关于静脉分数。** 0.617 是被*检出*而非*勾勒*所支配的：在**检出**了静脉的那 81.1% 的帧上，Dice 是 0.751，而同一判据下动脉是 0.905。静脉薄壁、在探头压力下会塌陷 —— 在最扁的那 20% 截面上召回率降到 0.46，且有 14 帧静脉被压至零截面，这是动脉从不出现的模式。Mus-V 的 105 次扫查只来自 11 名志愿者，这限制了单帧模型能学到多少这类形变。完整分析见 [results/unet/README_CN.md](results/unet/README_CN.md)。

---

## 部署

### 后处理

当前推理和评估共用 `models/unet/postprocess.py`：先在有单一类别锚点的血管区域修复局部动静脉混杂，再做半径2闭运算和填洞，过滤少于4像素的小区域，每类最多保留一个区域；不做开运算。补全不覆盖另一类保留区域，空输入保持为空。`min_area=1` 可保留极小目标。关键对照数值汇总保留；新规则的验证/测试结果见[后处理更新](results/unet/BASELINE_COMPARISON_CN.md)。

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
- **PMC9883282 颈内静脉接触力—塌缩数据** — [原论文与补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883282/)
- **Albumentations** — [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
- **Weights & Biases** — [wandb.ai](https://wandb.ai)
