# v11 单帧重训结果

2026-09-23。沿用五域 partition CE＋Dice，从原 v11 微调；本轮不使用时序模型。

**新候选未替换原 v11：PMC 静脉改善，但 Mus-V 静脉退化。**

[可浏览报告](generated/v11_refresh/index.html) · [Mus-V 原图、标注与新旧 v11 视频](generated/musv_review/index.html) · [训练日志](runs/v11_refresh_20260923/history.jsonl)

## 同条件比较

同一清理后划分、同一 joint 后处理，逐帧 Dice 平均；双空 Dice 记 1。历史测试已多次用于开发，不作为新的独立泛化证据。

| 划分 | 数据集 | 类别 | 帧数 | 原 v11 | 新候选 | 差值 |
|---|---|---|---:|---:|---:|---:|
| validation | musv | vein | 441 | 0.6092 | 0.5705 | -0.0387 |
| validation | musv | artery | 441 | 0.9126 | 0.9113 | -0.0013 |
| validation | pmc9883282 | vein | 28 | 0.3533 | 0.3730 | +0.0197 |
| validation | pmc9883282 | artery | 28 | 0.9471 | 0.9442 | -0.0028 |
| validation | mendeley | artery | 165 | 0.9589 | 0.9559 | -0.0030 |
| validation | phantom_taobao | vessel | 13 | 0.9251 | 0.9233 | -0.0018 |
| validation | customer_3d_phantom | vessel | 4 | 0.9640 | 0.9634 | -0.0006 |
| test | musv | vein | 470 | 0.6407 | 0.5934 | -0.0473 |
| test | musv | artery | 470 | 0.8928 | 0.8913 | -0.0015 |
| test | pmc9883282 | vein | 26 | 0.6917 | 0.7273 | +0.0356 |
| test | pmc9883282 | artery | 26 | 0.8526 | 0.8476 | -0.0049 |
| test | mendeley | artery | 165 | 0.9570 | 0.9536 | -0.0035 |
| test | phantom_taobao | vessel | 15 | 0.8682 | 0.8709 | +0.0026 |
| test | customer_3d_phantom | vessel | 5 | 0.9703 | 0.9693 | -0.0011 |

## 训练与局限

正式单卡 bfloat16 微调完成 5 个 epoch，验证选中 epoch 1。首次多卡非有限梯度运行已判无效，移至项目外历史归档，不计作训练收益。

PMC 标注从 362 张中排除 44 张无接触帧，保留 318 张（训练 264、验证 28、测试 26）。有效组织回声中的空血管标签仍保留。[数据审计](../../data/audits/20260923_contact_cleanup/contact_audit.json)。

本轮不仅增加标签，也改变了采样、增强和选模权重：PMC 训练采样占比约 19%→24%，验证选模权重 10%→30%。新增连续片段主要来自同一训练受试者。因此当前结果不能单独归因于新增标签好坏；下一次应固定其余设置比较旧/新标注，并在验证选模时限制各域退化。

![验证曲线](generated/v11_refresh/training.png)

## 文件与复现

- 模型代码：`models/unet/`；训练配置：`configs/v11_refresh.env`。
- 本地实验记录：`results/unet/runs/v11_refresh_20260923/`，整目录不上传 Git。
- [验证评估页面](runs/v11_refresh_20260923/validation_comparison/index.html) · [测试评估页面](runs/v11_refresh_20260923/test_comparison/index.html) · [替换决策](runs/v11_refresh_20260923/decision.json)。
- 活动权重仍是 `models/unet/checkpoints/unet_v11.pth`；新候选已归档，未覆盖默认。
- [时序模型结果](../temporal/RESULTS_CN.md) 单独放在 `results/temporal/`。

重新生成本摘要与页面：`python -m results.unet.summarize_refresh`。

原 v11 SHA256：`94ba45f9b1b6625c9a1c841e188dde52e866bbba42d96ade5e66ec12c119d9f4`。
