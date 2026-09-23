# 时序跟踪与预测：结果入口

当前模型为双尺度 ConvGRU-U-Net；候选未通过验收，原 v11 单帧基线保持不变。

- [实验结果、失败定位与优化方向](RESULTS_CN.md)
- [可浏览报告与训练曲线](generated/report/index.html)
- [完整 8fps 视频、逐帧控制与面积曲线](generated/video/index.html)
- [模型结构与运行方法](../../models/temporal/README_CN.md)
- [U-Net 单帧重训与 Mus-V 视频](../unet/V11_REFRESH_CN.md)

本轮本地记录：`runs/convgru_20260923/`，其中 `training/` 为最终训练，`evaluation/` 为统一对照，`provenance/` 为来源和迁移清单。`runs/` 与 `generated/` 整目录忽略，不上传。

重新生成报告：`python -m results.temporal.summarize`。重新生成视频：`python -m results.temporal.render_video --output results/temporal/generated/NEW_VIDEO`。脚本拒绝覆盖已有视频目录。

旧 Transformer、纠错支线及重复实验已从项目目录移出。仅留两次前置训练的简短曲线数据供本轮报告对照，不再保留旧模型入口。
