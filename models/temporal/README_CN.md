# Stage1：血管时序跟踪与预测

**双尺度 ConvGRU-U-Net 已实际训练和评估，本轮候选未通过验收。** 当前生产单帧权重仍是原 v11；这里的权重仅用于研究和复核，不宣称已解决跳变。

[完整数据处理、训练和结果](../../results/temporal/RESULTS_CN.md) · [可浏览报告](../../results/temporal/generated/report/index.html) · [完整 8fps 视频](../../results/temporal/generated/video/index.html)。

## 结构与监督

在原 U-Net 编码器 1/8、1/16 两个特征层插入 ConvGRU，隐藏通道 64，零初始化投影接回原特征；保留解码器和背景/静脉/动脉输出。每次读取一帧，在线更新两个隐藏状态。预测头从当前因果特征产生位移和有限概率调整，预测下一帧 mask，不使用未来图像。

最终训练固定 U-Net 权重、BatchNorm 和 Dropout 状态，只更新 ConvGRU 和预测头。8 个观测及下一帧目标均需人工完整 AV 标签：Mus-V 1571 个、PMC 73 个重叠窗口；不代表 1644 个独立样本或受试者。

`losses.py` 使用 full3 CE＋前景 Dice，分别监督动静脉中心和面积，并对相邻真实标签计算变化误差。背景、动脉、静脉分区在这两个数据集上均为完整监督；已有混合 partition loss 继续在五域 v11 训练中使用。

当前/未来/几何/变化项权重分别为 1/0.7/0.15/0.2。几何增强整段共享。中心变化只在对应类别两帧都存在时监督；面积变化保留出现/消失。无效超声帧切断状态，不把缺标帧当背景。

## 运行

```bash
# 从原始 Mus-V/PMC、当前 CSV 与接触审计重建，不依赖旧时序缓存。
python -m models.temporal.prepare --output /tmp/NEW_STAGE1_CACHE
# 复制 configs/temporal.json，改 cache/run_dir 到新路径后训练。
CUDA_VISIBLE_DEVICES=1 python -m models.temporal.train --config configs/YOUR_STAGE1.json
CUDA_VISIBLE_DEVICES=1 python -m models.temporal.evaluate \
  --checkpoint models/temporal/checkpoints/vessel_tracker_candidate.pth \
  --cache /tmp/uceeqz4_stage1_20260923 --output results/NEW_EVALUATION --joint
```

本轮缓存位于 `/tmp/uceeqz4_stage1_20260923`，可重建，不是原始数据。变更标签后必须重建缓存并记录新索引，旧 checkpoint 与旧评估不可冒充新标签结果。新 checkpoint 保存完整 U-Net 和时序参数。

```python
from models.temporal.infer import VesselTracker
tracker = VesselTracker('models/temporal/checkpoints/vessel_tracker_candidate.pth')
tracker.reset()  # 新受试者、新扫描或不连续跳转时必须重置
result = tracker.predict(cropped_grayscale_frame, dataset='pmc9883282')
# result['current'] 对应当前帧；result['future'] 对应下一原始帧步长。
# 明确无接触时 valid=False，两个输出为 None，同时清空隐藏状态。
```

输入为原生裁剪灰度图。对本地 PMC 使用经过看图复核的保守无接触规则，不把它当作通用设备接触分类器。输出回到输入图像网格。在线默认返回原始三分类标签；报告的 joint 指标额外使用相同的既有后处理。

## 本轮结论

v11 重训候选在 PMC 静脉上改善，但 Mus-V 静脉退化，未覆盖。Stage1 最终完成 6 个训练 epoch；没有训练权重通过验证门槛，研究候选是验证分数最高的实际训练 epoch 1。epoch0 仅为参考，没有作为训练成果发布。

去历史干预比完整历史在 Mus-V 上更好，说明本轮学到的状态更新存在负面影响；不能仅归因于标注数量。当前研究候选也未稳定超过复制和光流预测，后续重点应检查短片训练与长期递归状态的差异、当前/未来监督冲突，以及运动先验的作用。
