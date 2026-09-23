# 目录与 Git 约定

## 按功能归属，不按某轮 pipeline 新建顶层目录

| 内容 | 唯一位置 | Git |
|---|---|---|
| 单帧 U-Net 实现、训练、评估、推理 | `models/unet/` | 源码上传 |
| 当前双尺度 ConvGRU 时序实现 | `models/temporal/` | 源码上传 |
| 模型权重 | `models/<模型>/checkpoints/` | 不上传 |
| 单帧报告、绘图/视频生成代码 | `results/unet/` | 维护中的摘要与源码上传 |
| 时序报告、绘图/视频生成代码 | `results/temporal/` | 维护中的摘要与源码上传 |
| 日志、配置快照、逐帧指标 | `results/<模型>/runs/<实验名>/` | 整目录不上传 |
| HTML、视频、预测图集 | `results/<模型>/generated/` | 整目录不上传 |
| 数据处理源码 | `data/pipeline/` | 上传 |
| 原数据、标签、缓存、审计明细 | `data/datasets/`、`data/cache/`、`data/audits/` | 不上传 |
| 本地单元测试 | `tests/` | 不上传 |

`runs/` 和 `generated/` 忽略规则适用于任意层级，包括 `models/unet/runs/`。`generated/` 仅供本地展示，目录中的视频、页面、图片和数据文件都不上传。`test_*.py`、`*_test.py` 也忽略。正式评估代码已命名为 `evaluate.py`，不与单元测试混用。

PMC 的图像、人工标注和有效性标记统一放在 `data/datasets/PMC9883282/`。已合并的 `PMC9883282_S1Repair/` 重复队列已移除，选帧记录合入主目录原有的 `temporal_extension_v11.json`。无效帧通过主 metadata 的 `frame_valid=false` 排除，不为无效帧或每轮补标再建一份数据集。

## 本次整理

- `models/stage1/` 合并为当前 `models/temporal/`，旧时序源码移出项目。
- 混合的 `results/pipeline_20260923/` 拆入 U-Net、temporal 和数据审计对应位置，原目录撤销。
- Mus-V 新旧 v11 对照属于单帧结果，移到 `results/unet/generated/musv_review/`。
- 旧 Transformer 支线、重复尝试、旧视频及旧详细报告已归档到项目外 `../temporal_archive/20260923_layout_cleanup/`。归档不是活动模型入口，也不随项目上传。
- 必要复现记录保留在本地 `results/temporal/runs/convgru_20260923/provenance/`；历史日志、指标和权重内容不改写，旧路径可由迁移清单追溯。
- 已打开的 aligned_video、pmc_video 和旧 Mus-V 页面仅留小型跳转文件，不复制视频。

入口：[项目首页](../README_CN.md) · [单帧结果](../results/unet/README_CN.md) · [时序结果](../results/temporal/README_CN.md)。
