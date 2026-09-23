# U-Net

> 后续S1模型位于 [models/temporal](../../results/temporal/RESULTS_CN.md)，[首轮结果](../../results/temporal/RESULTS_CN.md)已完成；时序收益未通过验收，v11仍为对照。混合loss实现统一维护在 `models/losses.py`，本目录保留原训练接口。


> 跨模型的时序感知与 world model 设计见 [统一开发路线入口](../../results/temporal/RESULTS_CN.md)；本目录维护 U-Net 的实现、训练和推理说明。

> English: [README.md](README.md)

分割模型：一个输出**三通道 —— 背景、静脉、动脉**的 U-Net，用[顶层 README](../../README_CN.md) 里描述的部分标签目标函数训练。

这个架构需要的一切都在本文件夹里。新增架构就是在旁边加一个 `models/<arch>/`，放同样的五个文件；`data/` 下面一行都不用改。

| 文件 | 是什么 |
|---|---|
| [`model.py`](model.py) | 网络本体。`UNet(n_channels, n_classes, bilinear, base_ch)` |
| [`parts.py`](parts.py) | 构件（double conv、down、up、out） |
| [`train.py`](train.py) | 训练：部分标签 loss、验证、checkpoint |
| [`test.py`](evaluate.py) | 留出集评估、后处理、FPS |
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

`train.py` 默认读取仓库根的 [`.env`](../../.env)，也可通过 `--config` 指定独立实验配置。显式配置不合并根 `.env`；环境变量优先于文件配置。数据、checkpoint 路径以仓库根为基准。

```bash
python models/unet/train.py
```

通用模板为 [`configs/train.env`](../../configs/train.env)，默认原四域从零训练。无需为每个实验新增配置文件；用 `--set KEY=VALUE` 临时覆盖任意环境配置项，优先级为 **--set > shell 环境变量 > 配置文件 > 代码默认值**：

```bash
# 只查看最终参数及 from_scratch / finetune / resume，不加载数据、不训练
python models/unet/train.py --config configs/train.env --show-config

# 从零训练一个命名实验
python models/unet/train.py --config configs/train.env --set RUN_NAME=experiment_a

# 微调：仅加载权重，全新优化器/学习率；其余设置沿用通用模板
python models/unet/train.py --config configs/train.env --set \
  INIT_PATH=models/unet/checkpoints/unet_v10.pth RESUME_PATH= \
  LEARNING_RATE=0.0001 EPOCHS=100 RUN_NAME=finetune_a
```

`INIT_PATH` / `RESUME_PATH` 都为空表示从零训练；只设置 `INIT_PATH` 表示微调；只设置 `RESUME_PATH` 表示恢复 checkpoint 的训练状态，二者互斥。续训的 `EPOCHS` 是总目标 epoch，优化器/学习率状态从 checkpoint 恢复。

数据集组合通过 `DATASET`、`VAL_DATASET`、`VAL_WEIGHTS` 修改；也可设 `PRIMARY_VAL=mean` 等权评估。五域训练示例与研发进度见[主文档](../../docs/ultrasound_world_model.html#next-steps)。`dataPrepare.py --validate-only` 检查 `reviewed_csv` 的人工审核及图像/掩膜，不生成 NPZ；去掉此参数、使用 `--no-png` 生成缓存。模型 checkpoint 保存生效的训练配置。

**训练采样与验证权重分开设置。** `TRAIN_REPEATS=pmc9883282:50` 表示每张 PMC 训练原图每轮进行 50 次在线随机增强抽样；未指定的域沿用 `aug_times_train`。可用逗号同时调整多个域，例如 `TRAIN_REPEATS=pmc9883282:50,phantom_taobao:40`。次数必须是正整数，域必须包含在 `DATASET` 中。此设置不改 loss、增强强度或 val/test，不需要重建 NPZ；显式覆盖和运行时解析出的各域次数会写入 checkpoint。

当前五域重配比方案中，PMC 为 120×50＝6000 次，占 30,874 次/epoch 的 19.4%；其他四域次数不变。本轮 `unet_v11_balanced` 已完成从头训练，最佳权重归档为 `checkpoints/unet_v11.pth`，与 v10 同目录；旧中断实验及空子目录已清理。`VAL_WEIGHTS` 仍是模型选择权重，不是训练采样比例。

它会自己挑一张空闲 GPU（见 [`gpu_utils.py`](../../gpu_utils.py)），从 NPZ 缓存构建 dataloader，并把 checkpoint 写到 `CHECKPOINT_DIR`。

你实际会改的那些配置项：

| `.env` 键 | 默认值 | 作用 |
|---|---|---|
| `DATASET` | `musv+phantom_taobao+mendeley+customer_3d_phantom` | 训练集。用 `+` 连接即训练并集 —— 每个子数据集各自携带 `label_mode`，loss 会自动把每个样本路由到正确的监督模式 |
| `TRAIN_REPEATS` | 空，沿用数据集默认次数 | 按域覆盖每张训练原图每轮的在线抽样次数，例如 `pmc9883282:50`；不改变验证和测试 |
| `VAL_DATASET` | `musv,phantom_taobao+customer_3d_phantom,mendeley` | 验证集，逗号分隔。`+` 表示把两个合并成一个指标 |
| `VAL_WEIGHTS` | `phantom…:0.45, musv:0.45, mendeley:0.10` | 部署优先级，用于把各验证集分数合成 checkpoint 判据 |
| `NUM_CLASSES` | `3` | 3 = 背景 / 静脉 / 动脉。1 = 旧的二分类模型 |
| `BASE_CH` | `32` | Encoder 宽度。32 → 7.8 M 参数；64 → 31 M |
| `DROPOUT` | `0.3` | Dropout2d，只加在最深的两个 encoder 块和 bottleneck |
| `LEARNING_RATE` | `3e-4` | |
| `BATCH_SIZE_PER_GPU` | `16` | |
| `EPOCHS` / `EARLY_STOP_PATIENCE` | `200` / `30` | patience 必须远大于 `PLATEAU_PATIENCE`，否则降 LR 还没起效就停了 |
| `RUN_NAME` | `mixav_v3_cap32` | 会进入 checkpoint 文件名和 W&B run 名。**每次换实验前务必改**，否则文件名冲突 |
| `INIT_PATH` | 空 | 微调起点：只加载模型权重 |
| `RESUME_PATH` | 空 | 断点续训：恢复模型、优化器、scheduler 和 epoch；与 INIT_PATH 互斥 |
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
python models/unet/evaluate.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --dataset musv mendeley customer_3d_phantom phantom_taobao
```

按数据集逐个打表 —— 绝不把语义不同的数据集合成一个数字 —— 且后处理**前后**都打，这样后处理藏不住模型的真实行为。样例叠加图写到 `results/unet/predictions/<run>_<dataset>/`。

| 参数 | 默认 | 作用 |
|---|---|---|
| `--ckpt` | 必填 | checkpoint 路径 |
| `--dataset` | — | 一个或多个数据集。用 `+` 连接表示合并成一个指标，例如 `phantom_taobao+customer_3d_phantom` |
| `--min-area` | `4` | 联合类别修复、半径2闭运算和填洞后每类最多一个区域；可设1保留极小目标，不做开运算。 |
| `--max-dist` | `40` | 锚点（最大连通域）之外的碎片，离锚点超过这个距离就删（假阳性），更近的保留（同一根血管被切开了）。仅在 `--allow-fragments` 下生效 |
| `--allow-fragments` | 关 | 实验选项：允许保留附近碎片。默认每类最多一个区域，可以没有。 |
| `--vein-probe` | 关 | 在 `artery` 模式的数据集（mendeley）上探测模型预测了多少静脉。那里的静脉真实存在但没标注，训练和验证都完全不碰它，所以这是一个纯粹的泛化性检验 |
| `--samples` | 10 | 保存多少张叠加图 |
| `--no-save` | 关 | 不写叠加图 |
| `--no-fps` | 关 | 跳过吞吐测量 |
| `--batch` | — | batch size |

---

## infer.py —— 部署推理

自包含：只 import torch、numpy、cv2 和 `model.py` / `postprocess.py`，所以可以原样拷进机器人项目。没有训练侧 import，不读 `.env`，不碰数据集代码。

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
| `.predict(...)` | `(crop_frame, threshold=0.5, return_prob=False, postprocess=True, min_area=4, closing_radius=2, fill_holes=True)` | crop-frame 网格上的类别 id mask |
| `.predict_png(...)` | `(path, crop=None, **kw)` | 先读 PNG 的便捷封装 |
| `.measure(...)` | `(mask_crop, mm_per_px, axis_x=None, skin_row=0, min_area=1)` | `{"vein": {...}, "artery": {...}}`，含 `lateral_mm`、`depth_mm`、`radius_mm` |
| `.clean(...)` | `(binary, min_area=1)` | 保留最大8连通区域，空输入仍为空；不做开闭运算或填孔。 |
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

### 缺失血管、细线与显示约定

`predict()` 默认每类保留至多一个连通区域；`postprocess=False` 返回原始预测。`return_prob=True` 返回未经连通域筛选的血管概率。面积阈值在部署时以原图像素计，在 test.py 中以评估网格像素计。

`predict_regions(frame)` 返回 `vein`、`artery`、`vessel`（两者并集）、`overlap`（交集）、`display_rgb`（蓝静脉、红动脉、绿重叠）和 `vessel_rgb`（整个并集为绿）。当前 softmax argmax 互斥，交集通常为空；后续独立 mask 可用 `postprocess.av_outputs()` 合成。显示层 id=3 不可回写为人工标注或作为第四个模型输出。

全背景/仅动脉均为有效 full3 标注，沿用原混合部分标签 loss。连通域筛选不能保证去掉唯一一个假阳性，因此 test.py 额外报告 full3 缺失类别的假阳性帧率、像素数，以及全空帧的假阳性帧率（均越低越好）。这些诊断不改变 checkpoint 选择或原损失。原始与后处理 Dice 同时报告。样例另存绿色 `_vessel.png` 并集图。

默认150/300像素阈值及开运算已取消；当前联合后处理恢复小范围闭运算与填洞。历史结果需按当时协议复现，最新规则的验证/测试比较见下方更新。

## 双版本对照评估

当前保留 `checkpoints/unet_v10.pth` 与 `checkpoints/unet_v11.pth`，v11 为最佳 epoch 32（第 62 轮早停）。[逐步解读](../../results/unet/BASELINE_COMPARISON_CN.md)包含五域指标、空帧/缺失类误报和细静脉失败示例；预测对照图通过 `compare.py` 按需生成到Git忽略的 `results/unet/generated/`。

```bash
python models/unet/compare.py \
  --checkpoints models/unet/checkpoints/unet_v10.pth models/unet/checkpoints/unet_v11.pth \
  --postprocess joint --output results/unet/generated/v10_v11_test
```

从仓库根目录运行，输出目录须不存在。通用脚本支持两个任意版本的三类 checkpoint、`--datasets`、`--split` 和 `--batch`；原始预测与统一后处理均报告，不改变部分标签损失。最终权重每个版本保留一个 `unet_vN.pth`，与 v10 同目录；checkpoint 内原始实验配置保持可追溯。

## 联合后处理更新

当前 `predict()`、`predict_regions()`、`test.py` 与对照图使用共享 `postprocess.py`：对有单一主体类别证据的血管连通区域重归类（主体占比≥0.6），再做半径2的闭运算和填洞；少于4像素的独立区域删除，每类最多一个区域，允许为空。保留另一类主要区域，不对接触的两条血管强行投票合并。不做开运算；极细目标可用 `min_area=1`，`closing_radius=0, fill_holes=False` 可关闭形态补全。像素参数取决于处理网格，不能不经验证移植为毫米阈值。

`compare.py --postprocess joint`（默认）同时报告 raw / largest / cleaned，后者是新修复结果；`--postprocess largest` 可单独复现旧规则。静态 `clean(binary)` 仍是删除式单区域筛选，供几何阶段使用。新规则仍可能误归类或填多，不能保证分割完全正确。见[验证和测试结果](../../results/unet/BASELINE_COMPARISON_CN.md)。
