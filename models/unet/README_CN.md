# U-Net

> English: [README.md](README.md)

分割模型：一个输出**三通道 —— 背景、静脉、动脉**的 U-Net，用[顶层 README](../../README_CN.md) 里描述的部分标签目标函数训练。

这个架构需要的一切都在本文件夹里。新增架构就是在旁边加一个 `models/<arch>/`，放同样的五个文件；`data/` 下面一行都不用改。

| 文件 | 是什么 |
|---|---|
| [`model.py`](model.py) | 网络本体。`UNet(n_channels, n_classes, bilinear, base_ch)` |
| [`parts.py`](parts.py) | 构件（double conv、down、up、out） |
| [`train.py`](train.py) | 训练：部分标签 loss、验证、checkpoint |
| [`test.py`](test.py) | 留出集评估、后处理、FPS |
| [`infer.py`](infer.py) | **部署推理**。自包含 —— 只依赖 torch / numpy / cv2 + `model.py` |
| `checkpoints/` | 权重。不被 git 跟踪；`unet_v10.pth` 是已发布的那个 |

**已发布 checkpoint**：`models/unet/checkpoints/unet_v10.pth` —— `base_ch=32`，7 762 531 参数，dropout2d 0.3，epoch 19，best PRIMARY 0.8622。结果和完整实验记录在 [`results/unet/README_CN.md`](../../results/unet/README_CN.md)。

**所有命令都从仓库根目录运行。**

---

## Loss —— `partial_label_loss()`

数学形式和这个目标函数为什么存在，写在[顶层 README](../../README_CN.md)；每个数据集各自导出哪个划分，写在 [`data/README_CN.md`](../../data/README_CN.md)。本节讲的是**实现**，它在 [`train.py`](train.py) 里。

```python
partial_label_loss(logits,          # (B, 3, H, W) 原始 logits
                   target,          # (B, 1, H, W) long，id 在该样本自己的标签空间里
                   modes,           # 长度为 B 的列表："full3" | "vessel" | "artery"
                   class_weights=None,
                   cfg=None)
```

`log_softmax` 对整个 batch **只取一次**，然后按 `label_mode` 把 batch 切开，每组走自己的分支。一个 batch 里可以自由混合三种模式 —— `DATASET` 用 `+` 连接四个数据集时，这就是常态。

| `label_mode` | 前景 log 概率 | 背景 log 概率 | target 里允许出现的 id |
|---|---|---|---|
| `full3` | —（对全部通道做 3 类 `nll_loss`） | — | `{0 背景, 1 静脉, 2 动脉}` |
| `vessel` | `logsumexp(lp[:, [1, 2]])` = `log(p₁+p₂)` | `lp[:, 0]` = `log p₀` | `{0 背景, 3 未定类型血管}` |
| `artery` | `lp[:, 2]` = `log p₂` | `logsumexp(lp[:, [0, 1]])` = `log(p₀+p₁)` | `{0 背景, 2 动脉}` |

四个从公式上看不出来的实现要点：

**1. id 本身就是语义，路由错了是 assert 而不是容忍。** 每个分支都断言 target 里只含该模式允许的 id（`MODE_IDS`，在 [`data/pipeline/dataPrepare.py`](../../data/pipeline/dataPrepare.py)）。一个被路由到错误边缘概率的样本，会去监督错误的通道，而表面上只是「这个 epoch 效果差」—— 它永远不会报错。这个 assert 把静默的数据损坏变成崩溃。

**2. `full3` 的区域项只对血管类求平均，不是三类都算。** 三类 Dice 平均会被背景通道支配：背景占约 96% 的像素且永远接近完美，把它算进去会淹没掉 loss 真正要推动的静脉项。代码里 `_region_loss` 只对 `CLASS_VEIN` 和 `CLASS_ARTERY` 求平均。

**3. 全程留在 log 域。** `log(p₁+p₂)` 用 `logsumexp` 算，不是 `log(exp+exp)`，所以精确且在 AMP 下安全。两组模式下的 `log_bg = log(1 − p_fg)` 同样是直接读取真实通道，而不是用减法重构。

**4. 各模式按样本数加权合并，不是按组数。** 每个分支的 loss 乘以该模式在 batch 中的样本数，求和后除以 batch size —— 即样本数加权平均。一个恰好含 15 张 Mus-V 和 1 张仿体的 batch，会按这个比例加权。

### 区域项：Dice，还是 Tversky

`_region_loss` 默认是 `1 − Dice`。在 `.env` 里设 `TVERSKY_BETA` 会切换成 `1 − Tversky`，其中 `alpha` 给假阳性定价、`beta` 给假阴性定价（`alpha = beta = 0.5` 精确退化为 Dice）。

它存在是因为一个实测到的不对称：在 Mus-V val 上，**37% 的真静脉像素被判成了背景，而只有 1.3% 被判成动脉** —— 模型不是在混淆两种血管，而是直接漏掉了静脉。静脉薄、被压在组织上、对比度低，所以在一个把假阳性和假阴性定价相同的 loss（Dice 就是）之下，最安全的策略就是预测背景。`beta > alpha` 让漏掉一个静脉像素比凭空造一个更贵，而这正是问题本身具有的不对称性。**随仓库发布的 `.env` 里它是关闭的** —— 已发布的 checkpoint 用的是纯 Dice。

> `CE_CLASS_WEIGHTS` 同样是刻意不开的。在漏检的静脉像素上，`p₁` 中位数是 0.0019，而 `p₀` = 0.946；重新加权交叉熵够不到这么小的概率。

---

## train.py —— 训练

`train.py` **不接受任何命令行参数**。所有配置都从仓库根的 [`.env`](../../.env) 读取，所以一次运行完全由那个文件描述。

```bash
python models/unet/train.py
```

它会自己挑一张空闲 GPU（见 [`gpu_utils.py`](../../gpu_utils.py)），从 NPZ 缓存构建 dataloader，并把 checkpoint 写到 `CHECKPOINT_DIR`。

你实际会改的那些配置项：

| `.env` 键 | 默认值 | 作用 |
|---|---|---|
| `DATASET` | `musv+phantom_taobao+mendeley+customer_3d_phantom` | 训练集。用 `+` 连接即训练并集 —— 每个子数据集各自携带 `label_mode`，loss 会自动把每个样本路由到正确的监督模式 |
| `VAL_DATASET` | `musv,phantom_taobao+customer_3d_phantom,mendeley` | 验证集，逗号分隔。`+` 表示把两个合并成一个指标 |
| `VAL_WEIGHTS` | `phantom…:0.45, musv:0.45, mendeley:0.10` | 部署优先级，用于把各验证集分数合成 checkpoint 判据 |
| `NUM_CLASSES` | `3` | 3 = 背景 / 静脉 / 动脉。1 = 旧的二分类模型 |
| `BASE_CH` | `32` | Encoder 宽度。32 → 7.8 M 参数；64 → 31 M |
| `DROPOUT` | `0.3` | Dropout2d，只加在最深的两个 encoder 块和 bottleneck |
| `LEARNING_RATE` | `3e-4` | |
| `BATCH_SIZE_PER_GPU` | `16` | |
| `EPOCHS` / `EARLY_STOP_PATIENCE` | `200` / `30` | patience 必须远大于 `PLATEAU_PATIENCE`，否则降 LR 还没起效就停了 |
| `RUN_NAME` | `mixav_v3_cap32` | 会进入 checkpoint 文件名和 W&B run 名。**每次换实验前务必改**，否则文件名冲突 |
| `RESUME_PATH` | 空 | 要 finetune 的 checkpoint 路径。留空 = 从零训练 |
| `FREEZE_ENCODER` | `false` | 冻结 encoder；`UNFREEZE_EPOCH` / `UNFREEZE_LAYERS` 控制解冻 |
| `CHECKPOINT_DIR` | `models/unet/checkpoints` | 权重写到哪 |
| `MAX_GPUS` | `1` | 保持 1。多卡 DataParallel 是唯一一次观察到 val loss 变 NaN 的配置 |
| `WANDB_ENABLE` | `true` | 设 `false` 则不记录实验 |

也可以在命令行单次覆盖，不用改 `.env`：

```bash
RUN_NAME=expG_musv_3class DATASET=musv VAL_DATASET=musv \
python models/unet/train.py
```

**Checkpoint 选择**（`train.py` 的 `primary_score`）：主验证集若是全标注的（Mus-V），分数用 `(dice_vein + dice_artery)/2` —— 把两者分开**就是**任务本身，而合并的 `dice_vessel` 会在模型混淆动静脉时依然虚高；否则退回 `dice_vessel`。

---

## test.py —— 评估

```bash
python models/unet/test.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --dataset musv mendeley customer_3d_phantom phantom_taobao
```

按数据集逐个打表 —— 绝不把语义不同的数据集合成一个数字 —— 且后处理**前后**都打，这样后处理藏不住模型的真实行为。样例叠加图写到 `results/unet/predictions/<run>_<dataset>/`。

| 参数 | 默认 | 作用 |
|---|---|---|
| `--ckpt` | 必填 | checkpoint 路径 |
| `--dataset` | — | 一个或多个数据集。用 `+` 连接表示合并成一个指标，例如 `phantom_taobao+customer_3d_phantom` |
| `--min-area` | `0`（推荐 `150`） | 删掉面积小于此值的连通域。`0` = 关闭后处理，所以**要复现已发表的数字必须显式传 `--min-area 150`** —— 部署流水线是过滤的，而默认值不过滤。**别拿 Dice 去调它** —— val 上最优是 450，test 上最优是 100，两头顶天，纯粹在拟合噪声。按物理定：Mus-V 里最小的真静脉是 232 px，所以 150 保证删不掉真血管 |
| `--max-dist` | `40` | 锚点（最大连通域）之外的碎片，离锚点超过这个距离就删（假阳性），更近的保留（同一根血管被切开了）。仅在 `--allow-fragments` 下生效 |
| `--allow-fragments` | 关 | 关掉 keep-largest，改用距离门控。**默认是每类只留一个闭合区域**，因为真值里动脉 99.4%、静脉 90–95% 就是单连通域，而两个分离块算出的质心会落在两者之间的空隙里 —— 那是个不存在血管的位置，而它正是机器人的进针目标 |
| `--vein-probe` | 关 | 在 `artery` 模式的数据集（mendeley）上探测模型预测了多少静脉。那里的静脉真实存在但没标注，训练和验证都完全不碰它，所以这是一个纯粹的泛化性检验 |
| `--samples` | 10 | 保存多少张叠加图 |
| `--no-save` | 关 | 不写叠加图 |
| `--no-fps` | 关 | 跳过吞吐测量 |
| `--batch` | — | batch size |

---

## infer.py —— 部署推理

自包含：只 import torch、numpy、cv2 和 `model.py`，所以可以原样拷进机器人项目。没有训练侧 import，不读 `.env`，不碰数据集代码。

```python
from models.unet.infer import UNetInferencer

inf = UNetInferencer("models/unet/checkpoints/unet_v10.pth")

# mask 回到 CROP-FRAME 网格 —— 和你传进去的尺寸一致
mask = inf.predict(cropped_gray_frame)          # uint8: 0=背景, 1=静脉, 2=动脉

# 毫米制几何量，用当前深度档位对应的标定
g = inf.measure(mask, mm_per_px=0.0832)         # 4 cm 深度；5 cm 用 0.104
print(g["artery"]["lateral_mm"], g["artery"]["depth_mm"], g["artery"]["radius_mm"])
```

| 方法 | 签名 | 返回 |
|---|---|---|
| `UNetInferencer(...)` | `(ckpt_path, device=None, fit_mode="croppad")` | 加载 checkpoint，从里面存的 config 读出 `num_classes`、`base_ch`、`bilinear` |
| `.predict(...)` | `(crop_frame, threshold=0.5, return_prob=False)` | crop-frame 网格上的类别 id mask |
| `.predict_png(...)` | `(path, crop=None, **kw)` | 先读 PNG 的便捷封装 |
| `.measure(...)` | `(mask_crop, mm_per_px, axis_x=None, skin_row=0, min_area=300)` | `{"vein": {...}, "artery": {...}}`，含 `lateral_mm`、`depth_mm`、`radius_mm` |
| `.clean(...)` | `(binary, min_area=300)` —— 静态 | 开 → 闭 → 保留最大 → 丢弃小于 `min_area` |
| `.mm_per_px_from_depth(...)` | `(depth_cm, crop_h)` —— 静态 | 从扫描仪深度档位算标定常数 |

### 几何契约 —— 为什么 mask 要回到 crop frame

```
crop frame  --fit()-->  576x544  --模型-->  mask  --unfit()-->  crop frame
                                                                ^^^^^^^^^^
                                     几何量在这里测量，因为 mm_per_px 是在
                                     这个坐标系里定义的。模型从头到尾不知道
                                     毫米的存在。
```

`fit()` 只会做补边、中心裁剪，或者**把两个轴按同一个因子缩放** —— 宽高比从不改变 —— 而 `unfit()` 精确地把它逆回去。默认的 `croppad` 模式下根本不缩放，所以一个网络像素**就是**一个 crop-frame 像素，逆变换是纯偏移，零重采样损失。

这一点很重要，有两个理由，而且都是早期版本踩过的坑：

1. **非等比 resize 会把圆形血管变成椭圆。** 动脉 vs 静脉是一个**形状**判断 —— 动脉是圆的、不可压缩，静脉是扁的、可压缩 —— 所以两个轴按不同因子缩放，会毁掉三分类模型仅有的那个线索。
2. **几何量必须在 `mm_per_px` 被定义的坐标系里测。** 在 576×544 的网络空间里测量、再用为 crop frame 标定的常数换算，在两个轴被不同因子缩放过的情况下根本不成立 —— 没有任何一个标量能同时换算两个轴。在真实帧上实测，旧版本的**横向偏移偏了 +27.6%，半径偏了 +17.3%**。

`fit` / `unfit_mask` 在这里是**复制**的，而不是从 `data/pipeline/dataPrepare.py` import —— 这是刻意的：本文件必须没有任何训练侧依赖。它们必须和管线里的版本保持逐位一致。

---

## 相关

- [`../../results/unet/README_CN.md`](../../results/unet/README_CN.md) —— 结果、实验记录、论文素材
- [`../../analysis/unet_analysis/`](../../analysis/unet_analysis/) —— FPS 基准、几何测量、跨域探针、配置汇总
- [`../../data/README_CN.md`](../../data/README_CN.md) —— 数据集与预处理管线
