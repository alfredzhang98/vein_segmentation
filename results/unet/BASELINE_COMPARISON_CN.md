# 单帧基线对比：v10 → v11

本文件是本轮实验唯一维护的结果摘要；研究路线继续维护在 [主文档](../../docs/ultrasound_world_model.html)。Git 保留代码、关键指标和结论；PNG、HTML 图集、逐帧 CSV 和详细 JSON 按需重新生成到 `results/unet/generated/`，该目录不纳入 Git。

## 结论与模型

v11 在本轮相同测试集上，各域最终平均 Dice 均高于 v10，PMC 改善最大。但静脉跨受试者泛化、空帧误报和细静脉漏检仍未解决；v11 是五域单帧基线，不是时序 world model。

- v10：旧四域普通三类 U-Net，最佳 epoch 19。
- v11：五域从头训练，PMC 120 张训练图每轮重复50次，6000/30874≈19.4%；epoch62早停，保留epoch32最佳权重。现有部分标签混合损失不变。
- 权重同目录：`models/unet/checkpoints/unet_v10.pth`、`unet_v11.pth`。均为base_ch=32、背景/静脉/动脉三通道；权重和原始数据不纳入Git。

## 最终测试对比

两模型都使用**同一套联合后处理**。相同685张test图、相同缓存和归一化、无增强、AMP推理；Dice逐帧平均，双空记1。Mendeley只评价动脉，两个仿体只评价血管并集；不混合不同标注语义计算一个总Dice。

| 数据域 | 类别 | 帧数 | v10 最终 Dice | v11 最终 Dice | 差值 |
|---|---|---:|---:|---:|---:|
| musv | 静脉 | 470 | 0.6139 | 0.6407 | +0.0268 |
| musv | 动脉 | 470 | 0.8806 | 0.8928 | +0.0122 |
| musv | 血管并集 | 470 | 0.8054 | 0.8220 | +0.0166 |
| phantom_taobao | 血管并集 | 15 | 0.8667 | 0.8682 | +0.0016 |
| customer_3d_phantom | 血管并集 | 5 | 0.9694 | 0.9703 | +0.0009 |
| mendeley | 动脉 | 165 | 0.9558 | 0.9570 | +0.0012 |
| pmc9883282 | 静脉 | 30 | 0.3633 | 0.7328 | +0.3695 |
| pmc9883282 | 动脉 | 30 | 0.5947 | 0.8722 | +0.2776 |
| pmc9883282 | 血管并集 | 30 | 0.7768 | 0.9017 | +0.1249 |

## 后处理顺序与参数

1. argmax 得到背景、静脉、动脉；分别找面积至少4像素的最大8连通区域作为类别锚点。
2. 在原始血管并集的连通区域中，仅含一个类别锚点且主体占比≥60%时，将局部错色归回主体；不强行合并相接的动静脉主体。
3. 各类只保留面积至少4像素的最大区域，删除孤岛；允许为空。
4. 半径2的5×5椭圆核闭运算，再填封闭孔洞。不做开运算，空输入保持为空。
5. 不覆盖另一类保留像素；新增重叠按到补全前区域的距离归属，等距归静脉。
6. 再次保证每类最多一个区域，输出A/V与二者并集。仿体仅对并集清理和补全。

面积、半径按处理网格像素计；本轮固定min_area=4，极小目标可显式设1。评估在缓存输入网格，部署在原图网格，像素参数不能不经验证直接视为相同物理尺度。实现见 [postprocess.py](../../models/unet/postprocess.py)。

## 后处理收益与局限

固定同一v11权重，对比旧“每类只留最大区域，min_area=1”和新规则。先检查653张validation图，再对685张test图诊断，无参数搜索。

| 划分 / 类别 | 旧清理 Dice | 新修复 Dice |
|---|---:|---:|
| validation / pmc9883282 静脉 | 0.3909 | 0.3964 |
| validation / pmc9883282 动脉 | 0.9504 | 0.9506 |
| validation / musv 静脉 | 0.6080 | 0.6092 |
| validation / musv 动脉 | 0.9130 | 0.9126 |
| test / pmc9883282 静脉 | 0.7187 | 0.7328 |
| test / pmc9883282 动脉 | 0.8684 | 0.8722 |
| test / musv 静脉 | 0.6382 | 0.6407 |
| test / musv 动脉 | 0.8930 | 0.8928 |

- PMC源帧78的示例中，新规则能把静脉内的错红块、动脉内的错蓝块归回主体，避免旧规则直接删出缺口；主体类别都预测错时无法保证纠正。
- 并非所有样本/域都受益：v11 Mus-V动脉、customer phantom测试均值微降。新规则没有增加本轮v11缺失类误报帧数，也未消除原有误报。
- PMC test仅subject21的30帧（26张A/V、4张全背景），没有仅动脉帧；两版在这4张空帧均无误报，不能推广到所有人。Mus-V按序列划分，未证实患者完全隔离。
- v11最佳权重的**原始 validation**：Mus-V静脉0.6094，PMC静脉仍为0.3893（subject2）；PRIMARY复算0.837903，与保存值0.837898一致。不能用另一个人的较高test分数否认泛化缺陷。
- v10历史PRIMARY=0.8622使用不同域/权重，不能直接与v11的0.8379比较。
- 后处理受已浏览的test案例启发，本轮是探索性工程对照，不是完全独立的新泛化证明。一次训练结果未隔离数据增加、采样配比和训练随机性的影响。

## 按需生成完整结果

需要已下载的五域数据、已生成且指纹一致的缓存，以及两个权重文件；这些不随Git分发。缺少数据时按 [数据说明](../../data/README_CN.md) 准备。以下从 `vein_segmentation/` 运行：

```bash
conda activate ml_env
# 最终test对比：图片、HTML、逐帧CSV、详细JSON都写到Git忽略目录
python models/unet/compare.py \
  --checkpoints models/unet/checkpoints/unet_v10.pth models/unet/checkpoints/unet_v11.pth \
  --split test --postprocess joint --output results/unet/generated/v10_v11_test

# 同规则验证集检查
python models/unet/compare.py \
  --checkpoints models/unet/checkpoints/unet_v10.pth models/unet/checkpoints/unet_v11.pth \
  --split validation --postprocess joint --output results/unet/generated/v10_v11_validation
```

输出目录须不存在，以免覆盖已有实验。`compare.py` 支持任意两个三类checkpoint、`--datasets`、`--split`、`--batch`；不训练、不改标签。`joint` 报告 raw / largest / cleaned，最终口径为cleaned；`--postprocess largest` 可单独复现旧规则。默认生成完整PMC图集和其他域的代表/退步样例；历史手工拼接的封面及额外5张索引图不作为必需产物维护。

## 可追溯信息

以下SHA256来自实际评估输入，可在复现前核对。缓存指纹只检查配置一致，不能代替内容哈希。

| 输入 | SHA256 |
|---|---|
| unet_v10.pth | `032c243a7dffc99825e8b05a1cee28d3d295b2173400c8a9cf88cdd7a77579bf` |
| unet_v11.pth | `94ba45f9b1b6625c9a1c841e188dde52e866bbba42d96ade5e66ec12c119d9f4` |
| musv test缓存 | `28f7ad0afd97fe1b215a2873f4795b0ca7c313dfb60a82c52d0cc7427a595077` |
| phantom_taobao test缓存 | `a059d46348c504f1936e878385686f13c2115241dd6a09d0de831e74bcca3d29` |
| customer_3d_phantom test缓存 | `d84849744bfcb91a7a035bb62f8f4a8d47dd5940a150b35d4204e2a3499e8fb8` |
| mendeley test缓存 | `1fa3e487ad6dbc587940c838eb600f57f08573190fe13a91a7454a14f39c9eb4` |
| pmc9883282 test缓存 | `aa8dd247bc8b300d69d023009fcbb804d928a18595034910d0649bf0198d947e` |

后续进展：S1因果序列loader、模型与首轮训练/测试已完成，见[时序实验结果](../temporal/RESULTS_CN.md)。当前仅小幅修正，真实历史收益尚未通过验收，保留v10/v11对照，不自动切换部署权重。

## 当前展示图：每域两张，共八行、一个phantom

历史 `figures/segmentation_samples.png/.svg` 保留。本轮仅保留[最终八行展示图](figures/segmentation_samples_unet_v11_showcase.png)及同名前缀的选帧JSON，中间版本已移除。每个数据集展示两张（`--n 2`），同域两张共用一条灰色侧栏，与历史图一致。图中依次为Mus-V、Mendeley、PMC和Custom 3D phantom，使用最终联合后处理。Mus-V、Mendeley和PMC在每类Dice均不低于0.90的候选中兼顾质量和画面差异，phantom保持固定seed；全测试集统计仍包含两个phantom。

```bash
python results/unet/plot_segmentation_samples.py \
  --ckpt models/unet/checkpoints/unet_v11.pth \
  --datasets musv mendeley pmc9883282 customer_3d_phantom \
  --n 2 --showcase-datasets musv mendeley pmc9883282 --showcase-min-dice 0.90 \
  --out results/unet/figures/segmentation_samples_unet_v11_showcase
```

选帧先要求相关类别均在真值中可见，且每类Dice均≥0.90。第一张按最弱类别Dice择优，以均值打破平局；第二张在达标候选中选择与第一张差异最大的样本。差异为45%真值类别mask的Dice距离、45%各类别面积相对差异（面积差除以两者较大值）、10%归一化32×32图像的RMSE，兼顾血管位置、形状与图像外观；不保证来自不同受试者，PMC测试集仍只有一位受试者。空标签不能靠空集Dice满分入选。样式沿用历史 `segmentation_samples`：四列标题、灰色侧栏、大字号双列图例，无额外总标题或页脚。择优数据集、索引与分数记录在JSON；引用本图时应附上本段选帧说明。该图用于展示成功案例，不替代本页全测试集统计。

复现需要原始数据、测试缓存和模型权重。输出已存在则拒绝覆盖；再次生成请指定新的 `--out` 前缀。默认只生成PNG及选帧JSON，PDF/SVG需显式指定 `--formats`；最终 `segmentation_samples_unet_v11_showcase.png/.json` 可随仓库提交，其他生成产物仍由Git忽略。

当前选帧（test缓存索引；单帧Dice）：

| 数据集 | 索引 | 静脉 | 动脉 / vessel |
|---|---|---|---|
| musv | 335 | 0.9558 | 0.9553 |
| musv | 15 | 0.9384 | 0.9353 |
| mendeley | 17 | — | 0.9821 |
| mendeley | 79 | — | 0.9682 |
| pmc9883282 | 13 | 0.9660 | 0.9670 |
| pmc9883282 | 19 | 0.9203 | 0.9646 |
| customer_3d_phantom | 2 | — | 0.9740 |
| customer_3d_phantom | 4 | — | 0.9719 |
