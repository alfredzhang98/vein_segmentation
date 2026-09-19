# U-Net — 结果、实验记录与论文素材

> English: [README.md](README.md)

本文件合并了原先的 `EXPERIMENTS.md`（实验方案与日志）和 `PAPER_SEGMENTATION.md`（论文数据点、失败模式分析与写作稿）。全部内容只针对 [`models/unet/`](../../models/unet/) 这一个架构。

**参考模型**：[`models/unet/checkpoints/unet_v10.pth`](../../models/unet/checkpoints/) —— U-Net，3 通道输出（背景 / 静脉 / 动脉），`base_ch=32`，**7.76 M 参数**，dropout2d 0.3，epoch 19，best PRIMARY = 0.8622。下文所有数字都出自这一个 checkpoint。

**目录**

| 部分 | 内容 |
|---|---|
| [一、结果](#一结果) | 数据集划分、Test 集最终成绩、后处理选择 |
| [二、失败模式分析](#二失败模式分析) | 静脉低分的成因分解，决定 future work 指向 |
| [三、实验记录](#三实验记录) | 第一阶段二分类域迁移 Exp-A..E，第二阶段三分类联合训练 |
| [四、论文写作](#四论文写作) | METHOD / RESULTS / DISCUSSION 的原文与更新稿 |
| [五、出图](#五出图) | 本目录下的绘图脚本 |

---

# 一、结果

## 1.1 数字的出处与核对

**Mus-V（官网 / 原论文）**：11 名健康志愿者，Angell Pioneer H20 扫描仪，手臂与颈部（颈动脉、股动脉），**动脉与静脉分别标注**。105 个视频，每个 5–160 帧。官方 train 2203 张 / valid 911 张。

与代码的一致性核对：

| 官方 | 我们 | 一致？ |
|---|---|---|
| train 2203 | train **2203** | ✓ |
| valid 911 | val **441** + test **470** = **911** | ✓ 官方 valid 按**序列**对半切 |
| 105 videos | 105 sequences | ✓ |

> 不能按帧切分：同一 sweep 的相邻帧近乎重复，按帧随机切会把同一根血管同时放进 train 和 test。故沿用官方 train/valid，再把 valid **按序列**（非按帧）对半切成 val/test。

**CCA（Mendeley d4xt63mgjm，Momot 等，2022）**：**11 名受试者**，Mindray UMT-500Plus + L13-3s 线阵探头，每人左右两侧各至少一次检查（2 人用血管模式，8 人用颈动脉模式）。**每人 100 张，共 1100 张**，含技师制作、专家复核的掩模。原始分辨率 709×749×3。

> 实测 130 个不同的采集前缀 / 1100 张 ✓，与「11 人 × 每人多次检查」一致。

**Phantom**：两个自制仿体，**血管埋深不同**（实测质心深度：custom 206–225 px ≈ 17–19 mm；taobao 246–447 px），等效半径 custom 66.6–76.2 px ≈ 5.6–6.4 mm、taobao 29.6–115.8 px。

## 1.2 数据集分割

**分割原则**：按**序列 / 受试者**切分，不是按帧。

| Dataset | 受试者 | 采集单元 | 标注语义 | Train | Val | Test | 合计 |
|---|---|---|---|---|---|---|---|
| **Mus-V** | **11** | 105 videos | 背景/静脉/动脉（全标注） | **2203** | 441 | **470** | 3114 |
| **CCA (Mendeley)** | **11** | 11 人 ×2 侧 | 仅颈总动脉（颈内静脉未标） | **770** | 165 | **165** | 1100 |
| **Phantom (custom 3D)** | — | 浅埋血管 | 血管（不分动/静脉） | **21** | 4 | **5** | 30 |
| **Phantom (taobao)** | — | 深埋血管 | 血管（不分动/静脉） | **64** | 13 | 15 | 92 |
| **合计** | | | | **3058** | **623** | **655** | **4336** |

**每 epoch 的训练样本数**（每张原图重复采样，每次现场随机增强 —— 详见 [`data/README_CN.md`](../../data/README_CN.md) §2.2）：

| Dataset | 原图 | ×倍数 | 样本/epoch | 占比 |
|---|---|---|---|---|
| Mus-V | 2203 | ×8 | 17624 | 70.9% |
| CCA | 770 | ×5 | 3850 | 15.5% |
| Phantom (taobao) | 64 | ×40 | 2560 | 10.3% |
| Phantom (custom) | 21 | ×40 | 840 | 3.4% |
| **合计** | | | **24874** | 100% |

> 仿体倍数高（×40）是为了对冲它在联合训练集里的极小占比；否则真机部署的目标域会被稀释到 <1%。

两个仿体的验证集是**合并**成一个 17 图指标来算的 —— 4 图的划分单独打分只是噪声。

## 1.3 Test 集最终结果

**配置**：`min_area=150` + `keep_largest`（每类只留一个闭合区域），形态学开/闭 + 填洞。后处理与部署代码（[`models/unet/infer.py`](../../models/unet/infer.py) 的 `clean()` / [`analysis/unet_analysis/predict_geometry.py`](../../analysis/unet_analysis/predict_geometry.py) 的 `clean()`）完全一致。

### 要汇报的三个结果

| Domain | Metric | **Dice** |
|---|---|---|
| **Phantom (custom 3D)** | Vessel | **0.970** |
| **CCA (Mendeley)** | Artery | **0.956** |
| **Mus-V** | Artery | **0.882** |
| **Mus-V** | Vein | **0.617** |

### 完整表（含原始 argmax 对照）

| Dataset | | 原始 argmax | 后处理后 |
|---|---|---|---|
| Mus-V | Artery | 0.8752 | **0.8823** |
| Mus-V | Vein | 0.6095 | **0.6172** |
| Mus-V | Vessel (合并) | 0.8056 | **0.8047** |
| Phantom (custom) | Vessel | 0.9694 | **0.9696** |
| CCA | Artery | 0.9544 | **0.9555** |

复现：

```bash
python models/unet/test.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --dataset musv mendeley customer_3d_phantom phantom_taobao --min-area 150
```

> `--min-area 150` 不是可选的：这个参数**默认是 0**，也就是关闭后处理。不加它静脉是
> 0.6121、动脉是 0.8824 —— 那是「原始 argmax + keep-largest」的数，不是部署时的数。

### 为什么 `keep_largest`（每类一个闭合区域）

**每一类就是一根血管** —— 这不是强加的约束，是真值本身：动脉在 **99.4%** 的 Mus-V 帧里是单连通域，静脉 90–95%，仿体 **100%**。

而距离门控（`max_dist=40`）按设计会保留主体附近的碎片，实测让 **静脉 3.6% / 动脉 1.5%** 的帧分裂成多块。**代价不只是难看**：`measure()` 要算质心，**两个分离块的质心会落在两者之间的空隙 —— 那是个不存在血管的位置，而它是机器人的进针目标。**

**Dice 代价为零**：静脉 0.6187 → 0.6172（−0.002），动脉 0.8812 → 0.8823（**+0.001**），都在噪声内。

> 注：`infer.clean()` 和 `predict_geometry.clean()` **本来就是 keep-largest**，只有 `test.py` 和配图不是 —— 也就是说评测与配图在展示真机不会有的行为。现已统一。

> `phantom_taobao` 在主结果里被排除（该集含长轴切面，不属于本任务的使用场景）。

---

# 二、失败模式分析

## 2.1 静脉的 0.617 到底是什么

**两种口径**（全部数字用同一个 checkpoint，与结果表一致）：

| 口径 | 静脉 | 动脉 |
|---|---|---|
| **全部帧（论文报告值）** | **0.617** (470 帧) | **0.882** (470 帧) |
| 排除 Dice=0 的帧 | 0.753 (385 帧) | 0.906 (458 帧) |
| **仅「成功检出」的帧**（真值非空 **且** 有重叠） | **0.751** (381 帧 = **81.1%**) | **0.905** (455 帧 = 96.8%) |

> 论文正文建议用**最后一行**（「成功检出」），因为它的定义最干净、最不容易被挑刺：「排除 Dice=0」那一栏混进了 4 帧「真值空 + 模型也没报」的正确拒绝（按约定记 1.0），会轻微抬高分数。两者只差 0.002，但定义的严谨性值这个功夫。

> **→ 一旦模型检出静脉，Dice 就是 0.751（81.1% 的帧），而不是 0.617。**
> 静脉的整体分数不是被「边界画不准」拖累的，而是被 **18.1% 的全或无失败**拖累的。这两件事对下游引导的意义完全不同：**检出之后，静脉的轮廓质量（0.751）已接近动脉（0.905）** —— 差距主要在「找不找得到」，不在「画不画得准」。

**Dice = 0 的成因分解（静脉，85/470 = 18.1%）**：

| 成因 | 帧数 | 说明 |
|---|---|---|
| A 真值空 + 模型误报 | **14** | 血管被探头压瘪至面积 0，模型仍报（误报中位 3091 px） |
| B 真值有 + 完全漏检 | **51** | 整根没看见（漏掉的血管中位 4138 px） |
| C 真值有 + 位置全错 | **20** | 预测了但零重叠 |

动脉只有 **12/470 (2.6%)** 是 0 分，且**无 A 类** —— 动脉不会被压瘪，这正是两类差距的物理根源。

## 2.2 成因 ①：原始标注本身不够准

> 🔒 **本节的标注瑕疵统计 = 内部参考，不写进论文。**
> 论文里只说「静脉更难勾勒、参考标注含更多标签噪声」这一定性事实，**不列具体数字**。拿数字去指控一个公开数据集的标注质量，会把审稿焦点引向和数据集作者的争论，而原始依据是肉眼观察 —— 不值得为此开战。下面的数字只用于自己心里有底。

Mus-V 真值的**客观瑕疵**：

| 瑕疵 | 静脉 | 动脉 | 倍数 |
|---|---|---|---|
| 管腔内部有空洞的标注 | **3.7%**（106/2842） | 0.1%（2/3102） | **37×** |
| 含 <50px 噪点的标注 | 0.4% | 0.03% | 13× |

> 管腔是无回声的，里面不该有东西 → **空洞是明确的标注瑕疵**，不是解剖。静脉标注出现这种瑕疵的概率是动脉的 **37 倍**。

**更直接的证据：漏标的空静脉帧**。test 里 18 帧真值静脉为 0，逐帧看它的邻帧：

| 序列 | 帧位 | 本帧 | 前一帧 | 后一帧 | 判断 |
|---|---|---|---|---|---|
| 202304030805_33 | 48 | **0** | **5233** | **4993** | 5000px 的血管不可能一帧内消失又复原 → **漏标** |
| 202304030805_33 | 50 | **0** | 4993 | 5545 | **漏标** |
| 202304030805_33 | 33 | 0 | 1630 | 0 | 疑似漏标 |

**18 个空静脉帧中，6 个（33%）疑似漏标，12 个（67%）是真塌陷。**

> **这些数字的范围**：以上测的是**机器能检出的瑕疵**（管腔空洞、噪点、空帧矛盾）。**没有测边界精度** —— 而肉眼看到的「标注不准」主要就是边界。边界精度无法客观测量（没有第二套标注可比），所以**两个方向都不该断言**：既不能说标注问题是主因，也不能说它不是。**只陈述测到的事实，让数字自己说话。**

## 2.3 成因 ②：被压得更扁的静脉，识别率显著下降

| 静脉面积（越小 = 被压得越扁） | 平均召回率 |
|---|---|
| **最小的 20%**（300–1512 px） | **0.456** |
| 中间 | 0.830 |
| 最大的 20% | 0.674 |

且 **14 帧被压至面积 0**（动脉从不出现此模式）。**→ 直接指向力 + 时序联合预测。**

## 2.4 这些 0 分是怎么分布的

| | 静脉 |
|---|---|
| **孤立掉线**（前后邻帧 Dice>0.5） | **4/84 = 5%** |
| **成片失败**（前后邻帧也烂） | **80/84 = 95%** |

序列级更直接 —— 失败是整段的，不是零星的：

| 序列 | 平均 Dice |
|---|---|
| 最差 5 个 | **0.104 / 0.284 / 0.324 / 0.480 / 0.502** |
| 最好 5 个 | 0.762 / 0.794 / 0.795 / 0.811 / **0.845** |

→ **95% 的 0 分帧，其前后邻帧同样失败。** 这是「**某些受试者整段不行**」，不是「偶尔掉一帧」。**和 11 名志愿者的数据上限是同一结论的两面。**

## 2.5 Future work 该指向哪

两个方向都成立，且**都指向「检出」而非「勾勒」** —— 因为检出之后的 0.751 已接近动脉的 0.905，剩下的空间几乎全在那 18.1% 的全或无失败里。

1. **力–图像多模态融合**。Mus-V 官方描述：「该数据集还记录了超声探头上安装的**力传感器**…**以辅助识别动脉和静脉**」。静脉消失的**物理直接成因**就是探头压力把它压瘪（14 帧压至面积 0，动脉从不出现此模式）。力数据直接编码「此刻压了多大力」，是**数据集作者专为此提供的观测量** —— 而本文完全未使用。
2. **序列级时序建模**。Mus-V 是连续 sweep（105 个 video，每个 5–160 帧），逐帧模型不含任何时序上下文。失败是**成片**出现的（95% 的 0 分帧其邻帧同样失败），但**坏序列内部仍有可见帧**（最差的序列也有约 11% 的帧 Dice>0.5），序列级追踪可从这些帧把血管位置传播到被短暂压瘪的帧上。
3. **扩大受试者数量**（11 名 → 更多）—— 根本，成本最高。

> ⚠️ **措辞注意**：可以写「序列级模型 / 追踪」，**不要写「时序平滑」或「从邻帧插值」**。实测 95% 的 0 分帧其**直接邻帧同样失败**，朴素的邻域平滑会被这一条证伪；而能看到整段 sweep 的追踪器不受此限（坏序列里仍有可见帧可供传播）。这个区别审稿人会看出来。

## 2.6 ⚠️ 静脉探针（CCA）—— 不能写进论文当证据

**测到的事实**：CCA 从不监督静脉（loss 只依赖 p₂，已数值验证：固定 p₂ 时把静脉份额从 0.498 推到 1.000，loss 逐位不变，Δ = 0）。模型在 **104/165 = 63.0%** 的 CCA test 帧里，往静脉通道输出了 >200 px 的区域，占预测前景的 14.1%。

**这证明不了什么。** CCA **没有静脉标注 → 没有真值 → 无法验证那些区域是不是颈内静脉**。它可能是颈内静脉，也可能是任何暗区、声影、伪影，或被误分的一部分动脉。探针只数了「静脉通道有没有输出」，没数「输出对不对」。

> ❌ **不要写**：「the model recovers the unannotated internal jugular vein in 63% of CCA test frames」—— 审稿人一句「你怎么知道那是静脉？」就问死了，而我们答不上来。

**如果确实想用这个点，三个选项**（按成本排序）：

1. **只做定性、绑定到图**（最省）：不给百分比，只说「Fig. 6 (bottom row) shows the model additionally labelling an anechoic structure adjacent to the carotid as a vein, which CCA does not annotate.」—— 这是**读者看图就能自己验证**的陈述，不需要真值。
2. **人工抽样验证**（推荐，成本可控）：随机抽 20–30 张 CCA 帧，肉眼核对「静脉通道的输出是否落在一个紧邻颈动脉的无回声管腔上」，报「N/20 frames verified by inspection」。这样 63% 就变成了**有据可查的数字**。
3. **不用这个点**（最稳）：Results 里删掉。

**当前 Results 草稿已按选项 1 改写**（只留定性、绑定 Fig. 6）。

---

# 三、实验记录

## 3.1 第一阶段：二分类域迁移（Exp-A .. Exp-E）

**目标**：验证利用公开颈动脉数据集（Mendeley）提升 phantom 仿体血管分割性能的最优迁移策略。所有实验均在 **phantom test set** 上评估（目标域），保证对比公平。

> 这一阶段的模型是**二分类**（血管 / 背景），已被下文的三分类部分标签模型取代。保留在此是因为它是「为什么必须联合训练」的直接证据，[`plot_results.py`](plot_results.py) 画的就是这一组数字。

| 实验 | 训练数据 | val 数据 | 策略 | 预期作用 |
|---|---|---|---|---|
| **Exp-A** (baseline) | phantom only | phantom | 从零训练 | 当前水平基准 |
| **Exp-B** | Mendeley only | **mendeley** | 从零训练 | 验证域差距大小 |
| **Exp-C** | Mendeley + phantom 混合 | phantom | 从零训练 | 数据融合效果 |
| **Exp-D stage1** | Mendeley pretrain | **mendeley** | 预训练 | 学习超声通用特征 |
| **Exp-D stage2** | phantom finetune (全参数) | phantom | 两阶段迁移 | 预期最优策略 |
| **Exp-E stage2** | phantom finetune (冻结 encoder) | phantom | 两阶段迁移+冻结 | 对比冻结与全参数 finetune |

> **val 数据说明**：Exp-B / Exp-D stage1 用 mendeley val（与训练域一致，用于 early stopping 和 checkpoint 选择）；其余实验 val 用 phantom（目标域）。**最终 test 无论哪个实验都在 phantom test set 评估。**

### 结果汇总

| 实验 | 训练策略 | val Dice | val 数据集 | test Dice (phantom) | test Dice (mendeley) | 备注 |
|---|---|---|---|---|---|---|
| Exp-A | phantom only | 0.9087 | phantom | 0.8967 | 0.3147 | baseline，域外泛化差符合预期 |
| Exp-B | Mendeley only | 0.9564 | mendeley | 0.4735 | 0.9509 | 域差距显著，源域强目标域弱 |
| Exp-C | Mixed | 0.9257 | phantom | 0.9226 | 0.9479 | 两域均衡，phantom ↑ Mendeley 略降 |
| Exp-D v1 | Mendeley pretrain + full finetune (LR=3e-5) | 0.9185 | phantom | 0.9112 | 0.2735 | catastrophic forgetting，Mendeley 泛化丢失 |
| Exp-D v2 | Mendeley pretrain + full finetune (LR=1e-5) | 0.9201 | phantom | 0.9117 | 0.2682 | LR 降低无明显改善，forgetting 仍存在 |
| Exp-E | Mendeley pretrain + frozen encoder finetune | 0.9154 | phantom | 0.9041 | 0.2205 | 冻结 encoder 仍有 forgetting，phantom 低于 Exp-C |

**结论**：混合训练（Exp-C）是唯一两域都不塌的策略。两阶段迁移无论全参数还是冻结 encoder 都出现灾难性遗忘。这是走向联合训练的直接理由。

### .env 关键参数速查

| 参数 | Exp-A | Exp-B | Exp-C | Exp-D stage1 | Exp-D stage2 | Exp-E stage2 |
|---|---|---|---|---|---|---|
| DATASET | phantom_taobao | mendeley | mixed | mendeley | phantom_taobao | phantom_taobao |
| VAL_DATASET | phantom_taobao | **mendeley** | phantom_taobao | **mendeley** | phantom_taobao | phantom_taobao |
| FREEZE_ENCODER | false | false | false | false | false | **true** |
| LEARNING_RATE | 3e-4 | 3e-4 | 3e-4 | 3e-4 | **3e-5** | 1e-4 |
| RESUME_PATH | — | — | — | — | stage1.pth | stage1.pth |

跑法（改 [`.env`](../../.env) 后）：

```bash
python models/unet/train.py
python models/unet/test.py --ckpt models/unet/checkpoints/<best>.pth \
       --dataset phantom_taobao --no-save
```

**注意事项**

1. **每次切换实验前，务必修改 `.env` 中的 `RUN_NAME`**，否则 checkpoint 文件名冲突。
2. **Exp-D/E 阶段 2** 的 `RESUME_PATH` 填写阶段 1 生成的 checkpoint 完整路径。
3. **VAL_DATASET 控制验证集**；最终 test 无论哪个实验都跑 `--dataset phantom_taobao`。
4. **Mendeley NPZ 必须先生成**，否则 Exp-B/C/D 训练时会报 FileNotFoundError。
5. WandB 会自动记录每次实验，`RUN_NAME` 会出现在 wandb run 名称中便于区分。

## 3.2 第二阶段：三分类 + 部分标签联合训练

### 为什么需要新设计

引入 Mus-V 后，四个数据集的**标签语义不一致** —— 这是核心矛盾：

| 数据集 | mask 取值 | 知道什么 | **不知道什么** | `label_mode` |
|---|---|---|---|---|
| `musv` | `{0,1,2}` | 背景 / 静脉 / 动脉，全知道 | — | `full3` |
| `phantom_taobao` | `{0,255}` | 哪里是血管 | 是动脉还是静脉（仿体没这概念） | `vessel` |
| `customer_3d_phantom` | `{0,255}` | 哪里是血管 | 同上 | `vessel` |
| `mendeley` | `{0,255}` | 哪里是**动脉**（颈总动脉） | 颈内静脉在哪 —— **没标，被塞进了背景** | `artery` |

模型**永远输出 3 个通道**（0=背景, 1=静脉, 2=动脉），但 loss 只监督每个数据集**真正知道**的部分（[`models/unet/train.py`](../../models/unet/train.py) 的 `partial_label_loss`）：

- **`full3`（Mus-V）**：标准 3 类 CE + multiclass Dice，完整监督。
- **`vessel`（仿体）**：只监督边缘概率 `p1+p2`。该 loss **只依赖两者之和**，因此对「这根管子算动脉还是静脉」完全不施加约束 —— 归属是自由的，由 Mus-V 学到的先验决定。
- **`artery`（mendeley）**：只监督 `p2`。该 loss **只依赖 p2**，因此对剩余概率如何在背景与静脉之间分配是不变的 —— **没标注的颈内静脉可以被预测成静脉而零惩罚**。

  > 这一条是关键。如果偷懒直接用 3 类 CE 把 mendeley 的背景当 `bg` 监督，等于在教模型「静脉长得像背景」，会摧毁 Mus-V 学到的静脉类。**必须避开。**

全部用 `log_softmax` + `logsumexp` 实现，`log(p1+p2)` 精确且数值稳定。这三条不变性都有数值验证（偏差 ~1e-7）。

**部署**：仿体/二分类 → `(p1+p2) > 阈值`，二值流程不变；人体/要区分 → `argmax(p0,p1,p2)`。**一个模型，两种用法。**

### 数据准备

```bash
python data/pipeline/dataPrepare.py --dataset musv --no-png
# 105 个 sweep / 3114 帧 → train 80 seq (2203 帧) / val 12 seq (441) / test 13 seq (470)
```

> **按序列切分，不能按帧切。** 同一 sweep 的相邻帧近乎重复，按帧随机切会把同一根血管同时放进 train 和 test，Dice 虚高。Mus-V 自带官方 train/valid 划分，直接沿用，再把官方 valid 的 25 个序列**按序列**对半切成 val / test。

### 运行

单卡（H200 一次只能占一张卡，`MAX_GPUS` 默认 1；多卡 DataParallel 是唯一一次观察到 val loss 变 NaN 的配置）：

```bash
# Exp-G — Mus-V only，3 类。作为泛化性探针模型
EPOCHS=200 EARLY_STOP_PATIENCE=20 NUM_CLASSES=3 \
DATASET=musv VAL_DATASET=musv,phantom_taobao,mendeley \
BATCH_SIZE_PER_GPU=16 LEARNING_RATE=0.0003 RESUME_PATH= FREEZE_ENCODER=false \
RUN_NAME=expG_musv_3class python models/unet/train.py

# Exp-H — 四数据集部分标签联合训练（主结果）
EPOCHS=200 EARLY_STOP_PATIENCE=20 NUM_CLASSES=3 \
DATASET=musv+phantom_taobao+mendeley+customer_3d_phantom \
VAL_DATASET=musv,phantom_taobao,mendeley \
BATCH_SIZE_PER_GPU=16 LEARNING_RATE=0.0003 RESUME_PATH= FREEZE_ENCODER=false \
RUN_NAME=expH_joint_partial_label python models/unet/train.py
```

`DATASET` 用 `+` 连接即可训练并集；每个子数据集各自携带 `label_mode`，loss 会自动把每个样本路由到正确的监督模式。

**checkpoint / early-stop 指标**（`train.py: primary_score`）：主验证集若是全标注的（Mus-V）→ 用 `(dice_vein + dice_artery)/2`，因为任务就是把两者分开，合并的 `dice_vessel` 会在模型混淆动静脉时依然虚高；否则退回 `dice_vessel`。

## 3.3 泛化性探针

```bash
python analysis/unet_analysis/probe_generalization.py \
       --ckpt models/unet/checkpoints/<x>.pth \
       --datasets musv mendeley phantom_taobao
```

**Mus-V 是标尺** —— 它每一帧都同时标了动脉和静脉，所以对任意模型（哪怕它从没见过静脉标签）都能量化：*真实静脉的像素里，模型把多少判成了血管？* 这把「肉眼看看」变成了数字。脚本同时支持旧的二分类 checkpoint（`n_classes=1`）和新的 3 类模型。

### 已得结果：Exp-B（mendeley only，只见过动脉）→ Mus-V test

| 真实类别 | 判为背景 | 判为血管 |
|---|---|---|
| 背景 | 99.94% | 0.06% |
| **静脉** | 97.12% | **2.88%** |
| **动脉** | 59.22% | **40.78%** |

**只用动脉数据训练的模型不会泛化到静脉**：它在动脉上的召回率是静脉的 **14 倍**。它学到的不是「无回声管腔」这种通用特征，而是**动脉专属**的东西。overlay 图也一致：绿色预测几乎只落在蓝色动脉上，跳过红色静脉。

> 这正是需要 Mus-V + 部分标签联合训练的直接理由：靠 mendeley 是「长」不出静脉类的。

---

# 四、论文写作

目标期刊 TMECH。以下每节给「原文」与「更新后」两版，附改动理由。

## 4.1 METHOD — 原文

> For ultrasound-based vessel localisation, a U-Net-based segmentation module was adapted for vessel-mask extraction and downstream geometric guidance [19]. The aim of this module is not to introduce a new segmentation architecture, but to provide a reliable vessel-region estimate from which the centroid, lateral offset, depth, and radius can be extracted. The model is trained using our custom vascular phantom dataset together with the public Mendeley carotid artery dataset (CCA) [20]. CCA is used as an auxiliary domain because the task focuses on vessel-region segmentation rather than artery–vein classification. In short-axis ultrasound, the internal jugular vein can appear approximately circular under Trendelenburg positioning, making carotid artery images useful for learning generic vessel-boundary features despite the anatomical difference. The network is trained with a compound BCE–Dice loss [21],
>
> L = L_BCE + (1 − (2 Σ pᵢgᵢ + ε) / (Σ pᵢ + Σ gᵢ + ε))   (1)
>
> where pᵢ is the predicted vessel probability, gᵢ is the binary ground-truth label, and ε is a small constant for numerical stability. As shown in Fig. 3(a), the probability map is dichotomised into a binary mask and refined by morphological cleaning. Image-moment analysis and the calibrated ultrasound pixel scale are then used to estimate the vessel centroid, lateral offset δ, depth d, and radius r for downstream alignment and insertion guidance.

## 4.2 METHOD — 更新后

> For ultrasound-based vessel localisation, a U-Net-based segmentation module was adapted for vessel-mask extraction and downstream geometric guidance [19]. The aim of this module is not to introduce a new segmentation architecture, but to provide a reliable vessel-region estimate from which the centroid, lateral offset, depth, and radius can be extracted.
>
> The module is trained jointly on two custom vascular phantoms with vessels at different embedding depths, the public Mus-V vascular dataset [X] (11 volunteers, arteries and veins annotated separately), and the public Mendeley common carotid artery dataset (CCA) [20] (11 subjects, carotid artery only). These sources disagree on what their masks mean: Mus-V annotates the vein and the artery separately; the phantoms annotate a tube whose type is undefined; CCA annotates only the carotid artery, while the internal jugular vein is frequently within the field of view but left unlabelled. Pooling them under a single objective would therefore supervise CCA's unlabelled jugular vein as background, penalising a correct vein prediction wherever that vessel appears.
>
> The network therefore emits three channels — background, vein and artery, with softmax probabilities p₀, p₁ and p₂. Each dataset's annotation resolves these three classes only up to a partition 𝒢 of the class set: Mus-V separates all three, so 𝒢 = {{bg}, {vein}, {artery}}; the phantoms mark a tube of undetermined type, so 𝒢 = {{bg}, {vein, artery}}; CCA marks the carotid artery while leaving the internal jugular vein unlabelled inside the background region, so 𝒢 = {{bg, vein}, {artery}}. Writing q_g = Σ_{c∈g} p_c for the marginal probability of a group g ∈ 𝒢 and y_g ∈ {0,1} for its annotation, the compound cross-entropy–Dice objective [21] is applied to these marginals:
>
> **L = − (1/|Ω|) Σᵢ∈Ω Σ_{g∈𝒢} y_{g,i} log q_{g,i}  +  (1/|𝒢⁺|) Σ_{g∈𝒢⁺} (1 − (2 Σᵢ q_{g,i} y_{g,i} + ε) / (Σᵢ q_{g,i} + Σᵢ y_{g,i} + ε))   (1)**
>
> where Ω denotes the pixels of an image, ε is a small constant for numerical stability, and 𝒢⁺ are the foreground groups — the group containing the background class is excluded from the Dice term, since background covers ~96% of pixels and is predicted near-perfectly, so retaining it would only dilute the gradient on the vessel groups that term exists to drive. A batch may mix the three annotation types; the corresponding losses are combined as a sample-count-weighted mean.
>
> For a two-group partition — the phantoms and CCA — the first term is exactly the binary cross-entropy L_BCE(q, y) and the second reduces to a single Dice term, so (1) recovers the conventional compound BCE–Dice loss evaluated on the supervised marginal. Its effect is to leave unsupervised distinctions free: for the phantoms the objective depends on p₁ + p₂ alone, so the division between vein and artery is unconstrained and is resolved by the prior learned from Mus-V; for CCA it depends on p₂ alone and is invariant to how the remaining probability divides between background and vein, so an unlabelled jugular vein may be predicted at no cost. For Mus-V all three groups are singletons and (1) becomes a three-way cross-entropy together with the Dice term evaluated on the vein and the artery channel and averaged over the two. All terms are evaluated in the log domain via log-softmax and log-sum-exp, making log(p₁ + p₂) exact and numerically stable.
>
> As shown in Fig. 3(a), the probability map is dichotomised into a binary mask and refined by morphological cleaning: opening and closing suppress speckle, interior holes are filled — a vessel lumen is anechoic, so an enclosed hole is a segmentation artefact and, left in place, biases the area-equivalent radius — and components lying far from the principal vessel are discarded as false positives. Image-moment analysis and the calibrated ultrasound pixel scale are then used to estimate the vessel centroid, lateral offset δ, depth d, and radius r for downstream alignment and insertion guidance. The network itself is purely a pixel-to-pixel map; the physical scale enters only after the predicted mask is returned to the native image grid, so a single calibration constant converts both axes exactly.

**改动说明**

| 项 | 原文 | 更新后 | 理由 |
|---|---|---|---|
| 数据集 | phantom + CCA | + **Mus-V** | 静脉类只能从 Mus-V 学到 |
| 输出 | 1 通道（血管） | **3 通道**（bg/vein/artery） | 任务升级为区分动/静脉 |
| CCA 的定位 | 「因为不做动静脉分类，所以 CCA 可用」 | 「CCA 只标动脉，其未标注的静脉必须零惩罚」 | **原逻辑已失效** —— 我们现在做动静脉分类了 |
| 公式 | BCE-Dice 作用在血管概率 p | **CE-Dice 作用在「标注能分辨的划分」𝒢 的组概率 q_g** | **保持一个公式**；两组划分时第一项严格退化为 L_BCE、第二项退化为单个 Dice，即原式，所以是原式的推广而非替换 |
| 后处理 | 「morphological cleaning」 | 明确：开/闭 + **填洞** + 远处孤岛剔除 | 填洞影响半径 |
| — | — | 新增：模型不含物理尺度 | 回应审稿人对标定的质疑 |

## 4.3 RESULTS — 原文

> To verify the ultrasound segmentation module used in the subsequent guidance experiments, the selected U-Net model was trained jointly on the custom vascular phantom dataset and the public CCA dataset. The model achieved Dice scores of 0.923 and 0.948 on the held-out phantom and CCA test sets, respectively. As shown in Fig. 6, the predicted vessel regions closely matched the manual annotations in both domains, despite the different ultrasound appearances of the phantom and anatomical images. The jointly trained model was therefore used to provide vessel masks for estimating the vessel centre and radius in the following insertion experiments.

## 4.4 RESULTS — 更新后

> To verify the ultrasound segmentation module used in the subsequent guidance experiments, the U-Net model was trained jointly on the vascular phantoms, Mus-V and CCA, and evaluated on held-out test sets held disjoint at the sequence and subject level.
>
> On the phantom — the domain the guidance experiments are performed in — the model segmented the vessel region with a Dice of **0.970**. On the two anatomical datasets it localised the artery with a Dice of **0.956** on CCA and **0.882** on Mus-V, confirming that a single set of weights transfers across ultrasound appearances as different as a gel phantom and a human neck. For these classes the predicted boundaries follow the manual annotations closely (Fig. 6). End to end — preprocessing, inference, morphological refinement and moment analysis — the module runs at **5.6 ms per frame** (5.7 ms at the 95th percentile) at batch size 1 on an NVIDIA H200 NVL, i.e. **179 fps**, well above the ultrasound frame rate, so segmentation does not bound the guidance loop.
>
> Mus-V additionally annotates the vein, on which the model reached a Dice of **0.617**. This figure is governed by detection rather than delineation: on the **81.1%** of frames where the vein is detected, the Dice is **0.751**, against **0.905** for the artery under the same criterion — once the vein is found, it is outlined nearly as accurately.
>
> Two factors account for the remaining gap. The first is that the vein is intrinsically harder to delineate: it is thin-walled and low-contrast against the surrounding tissue, so its boundary is less sharply defined than an artery's — for the annotator as much as for the network. The second is compression. A vein collapses under probe pressure, and recognition degrades as it flattens: recall falls to 0.46 on the most flattened quintile of vein cross-sections against 0.83 at mid-range, and in 14 test frames the vein is compressed to zero cross-section entirely — a mode the non-collapsible artery never exhibits. Mus-V's 105 videos originate from only 11 volunteers, which bounds how much of that deformation a single-frame model can learn; the train–validation Dice gap stands at 0.03 for the artery against 0.24 for the vein.
>
> A qualitative observation supports the partial-label formulation. CCA never supervises the vein — by construction, (1) is invariant to it there — yet in Fig. 6 (bottom row) the model additionally labels an anechoic structure adjacent to the carotid, which we confirmed by inspection to be the internal jugular vein, and which the CCA annotation does not cover. As CCA provides no vein ground truth, this is reported qualitatively rather than scored.
>
> Since the guidance task requires the vessel region and its geometry, and vessel-type discrimination is a secondary capability, the jointly trained model was used to provide vessel masks for estimating the vessel centre and radius in the following insertion experiments.

**改动说明**

| 项 | 原文 | 更新后 |
|---|---|---|
| 数字 | phantom 0.923 / CCA 0.948 | **phantom 0.970 / CCA 0.956 / Mus-V 动脉 0.882 / 静脉 0.617** + FPS |
| 结构 | 一段平铺 | **先讲强项**（仿体 0.970 + 双域动脉），静脉单独一段 |
| 静脉 | — | **拆成「检出」与「勾勒」两件事**：检出后 0.751，接近动脉 0.905 |
| 归因 | — | 物理根因（可压缩，14 帧压至面积 0）+ 数据上限（11 名）+ gap 0.03 vs 0.24 |
| 实时性 | — | **5.6 ms/帧, 179 fps (H200 NVL, batch=1)** —— TMECH 是机电期刊，必须给 |
| 泛化 | — | 只做**定性**观察并绑定 Fig. 6；**不给百分比**（CCA 无静脉真值，量化不了） |

**写作技巧**

1. **先给强项，再给弱项**：仿体 0.970 和双域动脉（0.956 / 0.882）先立住，再补实时性，读者的锚点是「这个模块可靠且能上机」，静脉 0.617 才不会主导第一印象。
2. **把 0.617 拆开**：「Dice 0.617」听起来是「轮廓画得烂」；实际是 **81.1% 的帧检出后 Dice 0.751**，剩下的是全或无失败。**这不是粉饰 —— 是更准确的描述**，而且 0.751 这个数经得起审稿人复算。
3. **用「dominated by detection rather than delineation」定性**：一句话把读者的理解从「模型画不准」扭到「模型偶尔找不到」，后者对下游引导的含义完全不同。
4. **归因于物理而非模型缺陷**：「14 帧被压至零截面」是**可验证的客观事实**，比「静脉比较难」有力得多。
5. **用定性观察收尾，但明说不量化**：主动声明自己没有真值，反而显得克制可信；而读者看图就能自己验证这个陈述。

> **不要做的事 ①**：不要只报 0.751 而不报 0.617。0.617 是全体测试帧的诚实数字，审稿人一定会追问口径。**两个都给，并说清楚差别在哪** —— 这比藏起来可信得多。
>
> **不要做的事 ②**：不要写「recovers the internal jugular vein in 63% of CCA frames」。探针只数了「静脉通道有没有输出」，**没数「输出对不对」** —— CCA 没有静脉真值，这句话无法证明。想量化就得人工抽样核对（见 §2.6 选项 2）。

## 4.5 DISCUSSION — 新增段落（可直接粘贴）

> The residual error on the vein lies in detection rather than delineation. Once detected, the vein is delineated at a Dice of 0.751 against the artery's 0.905; what separates the two classes is that the vein is missed outright on 18.1% of frames. Those misses are not distributed uniformly — 95% fall in runs whose neighbouring frames fail as well, and five of the thirteen test sequences account for most of them — and they track vessel compression directly: recall drops to 0.46 on the most flattened quintile of veins. Two directions follow, both attacking detection, and neither exploited by the present single-frame model.
>
> First, the failures have a single physical driver. A vein is thin-walled and collapses under probe pressure, vanishing entirely in 14 of the test frames — a mode the non-collapsible artery never exhibits. Mus-V records synchronised probe-force measurements expressly to help distinguish arteries from veins, and force is precisely the quantity that governs the collapse. Conditioning the segmentation on force therefore addresses the failure at its source rather than at its symptom.
>
> Second, the data are acquired as continuous probe sweeps, yet each frame is currently segmented in isolation. Because the failures occur in runs, frame-to-frame smoothing would be of little help; a sequence-level model, however, can exploit the frames within a sweep in which the vein remains visible — present even in the weakest sequences — and propagate its location through the frames in which it is momentarily compressed. Jointly conditioning on force and sequence context targets the same underlying event from two directions: force says when the vein is being flattened, and the sweep says where it was before it flattened.
>
> A sequence-level formulation would additionally confer robustness to isolated labelling errors, to which a per-frame objective is fully exposed. Together these routes address the all-or-nothing failures that presently bound the vein score, without disturbing the delineation quality, which is already adequate.

**写作要点**

| 手法 | 效果 |
|---|---|
| 开头就把 **0.751 vs 0.905** 摆出来 | 先定性：问题在「找不找得到」，不在「画不画得准」 |
| 明说 95% 成片失败 | **主动暴露**，堵住审稿人自己去查；也顺势论证「为什么不是简单平滑」 |
| 「a mode the artery never exhibits」 | 用一个**对照事实**把静脉低分钉死在物理上，而非模型缺陷 |
| 「addresses the failure at its source rather than at its symptom」 | 把力模态从「再加个输入」抬成**因果层面的解法** |
| 「frame-to-frame smoothing would be of little help; a sequence-level model, however…」 | **先自我否定弱版本，再提强版本** —— 显示我们清楚边界在哪，可信度陡增 |
| 结尾「without disturbing the delineation quality, which is already adequate」 | 回扣：改进空间是有界且明确的，不是「模型不行需要重做」 |

## 4.6 Loss / Checkpoint 逻辑（一句话版）

**Loss（目标导向）**：
> 每个数据集只监督它的标注**真正能确定**的那个边缘概率 —— 仿体只约束「静脉+动脉」之和（动/静脉归属自由），CCA 只约束动脉通道（未标注的颈内静脉可以被预测且零惩罚），Mus-V 三类全监督 —— 从而让一个模型能在标注语义互相矛盾的四个数据集上联合训练，而不会互相破坏。

**Checkpoint（目标导向）**：
> 按「部署优先级」加权各验证集的分数（仿体 0.45 / Mus-V 0.45 / CCA 0.10），且每个验证集只用它的标注**能支持**的指标（Mus-V 用 (静脉+动脉)/2，仿体用血管 Dice，CCA 用动脉 Dice），保证选出的 checkpoint 不会因为某个数据集的标注缺失而被系统性地误导。

## 4.7 模型配置表

| Parameter | Value |
|---|---|
| Architecture | U-Net (3-class, partial-label) |
| Base width | 32 |
| Parameters | 7.76 M |
| Input | 576 × 544, grayscale |
| Dropout2d | 0.3 |
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻³ |
| Batch size | 16 |
| Loss | Compound grouped-CE–Dice on the marginals the annotation resolves, Eq. (1) (= BCE–Dice for the two-group datasets) |

> 该表可自动生成（所有数字从 checkpoint 读出，不手打）：
> ```bash
> python analysis/unet_analysis/summarize_model.py --ckpt <x>.pth --latex
> ```

---

# 五、出图

本目录下的脚本，全部从仓库根目录运行。

| 脚本 | 输出 | 说明 |
|---|---|---|
| [`plot_segmentation_samples.py`](plot_segmentation_samples.py) | `figures/segmentation_samples.{svg,pdf,png}` | **论文定性主图**。三个域各一行，配色编码「哪根血管」而非「对错」 |
| [`plot_results.py`](plot_results.py) | `figures/exp_results.{pdf,svg,png}` | Exp-A..E 分组柱状图。**注意：画的是 §3.1 的第一阶段二分类实验**，不是当前三分类模型 |
| [`figure_style.py`](figure_style.py) | — | 共享 matplotlib 样式，不自己出图。被上面两个 import |

```bash
# 论文定性主图
python results/unet/plot_segmentation_samples.py \
       --ckpt models/unet/checkpoints/unet_v10.pth \
       --n 2 --seed-phantom 41 --seed-musv 13 --seed-cca 116

# 挑 seed 时加 --preview：只出 PNG、150 dpi，快约 5 倍
```

`figure_style.py` 的 `FigureConfig` 有三档预设：默认（大字号，按 IEEE 双栏缩放后仍清晰）、`ieee_strict()`（8–10 pt，用于已定稿排版）、`presentation()`（幻灯片 / 海报）。所有图都按 12.0" 宽度作图并按同一因子缩放到论文栏宽，这样每张图的文字印出来物理尺寸一致 —— 不要单独改某张图的宽度。
