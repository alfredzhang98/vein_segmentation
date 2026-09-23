# PMC9883282 视频预览

本目录只在 Git 中保留 `generate_preview.py` 和本说明。MP4、WebM、`index.html` 和 `manifest.json` 是可重新生成的本地产物，已加入 `.gitignore`；已有视频可以继续本地使用。

采集部位为左侧颈部，以左颈内静脉短轴横截面为目标，旁边可见颈动脉。研究观察探头接触力与颈内静脉压缩形变。来源：[PMC9883282 原论文及补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883282/)（Jaffe 等，2023，CC BY 4.0）；实验条件依据数据附带的 `readme.txt` 和 MAT 文件名。

## 1. 先检查数据是否已经下载

从 `vein_segmentation/` 根目录执行，使用已有环境，无需安装服务器系统软件：

```bash
conda activate ml_env
python data/previews/PMC9883282/generate_preview.py --check-only
```

默认读取 `data/datasets/PMC9883282/`。路径以脚本所在位置确定，因此从其他工作目录调用也可以。脚本检查六个原始 MAT 是否齐全、能否打开，以及 `ultrasound_images` 的类型、维度和首尾帧是否可读。缺失时会列出文件名并退出，不会自动下载或写入预览。此检查不是整个下载包的校验和认证；生成时还会逐帧读取数据。

预期文件名为下表中的名称加上 `_prepped_force_ultrasound.mat`，全部数据合计 **5215 帧**：

| 名称 | 条件 | 帧数 | 默认 20 fps 预览时长 |
|---|---|---:|---:|
| subject20_16degree | 受试者20：抬高16度、逐渐加压 | 548 | 27.40 秒 |
| subject20_Valsalva | 受试者20：Valsalva动作、逐渐加压 | 637 | 31.85 秒 |
| subject20_constant | 受试者20：仰卧、恒定接触力 | 1463 | 73.15 秒 |
| subject20_supine | 受试者20：仰卧、逐渐加压 | 588 | 29.40 秒 |
| subject21_supine | 受试者21：仰卧、逐渐加压 | 756 | 37.80 秒 |
| subject2_supine | 受试者2：仰卧、逐渐加压 | 1223 | 61.15 秒 |

如果没有下载，打开论文的 **Supplementary Information / Supplementary Data**，下载包含上述 MAT 的数据包。也可使用现有下载工具获取：

```bash
curl -fL --retry 3 \
  https://pmc-oa-opendata.s3.amazonaws.com/PMC9883282.1/41598_2022_22867_MOESM1_ESM.zip \
  -o /tmp/PMC9883282_supplement.zip
python -m zipfile -e /tmp/PMC9883282_supplement.zip /tmp/PMC9883282_raw
```

找到解压后**直接包含六个 MAT 文件**的目录，用 `--data-dir /实际路径` 指定；或者将这些 MAT 和原始 `readme.txt` 放到 `data/datasets/PMC9883282/` 后重新检查。不要用下载包覆盖已有的 `images/`、`masks/`、标注 CSV 或抽帧配置。若论文补充材料链接变动，以论文页面的数据包为准。

```bash
python data/previews/PMC9883282/generate_preview.py \
  --data-dir /实际路径/包含MAT的目录 --check-only
```

当前工作区已核对六个 MAT 齐全；新克隆的仓库不包含数据，因此仍需自行检查。数据目录可能是软链接，换机器后需要恢复其目标或显式指定 `--data-dir`。

## 2. 生成或重新生成预览

```bash
# 首次生成：全部六段，同时输出 MP4 和 WebM
python data/previews/PMC9883282/generate_preview.py

# 已有预览时，显式替换生成产物
python data/previews/PMC9883282/generate_preview.py --overwrite

# 只生成某一段的浏览器视频，放在独立目录
python data/previews/PMC9883282/generate_preview.py \
  --sequence subject20_supine --format webm \
  --output-dir /tmp/pmc_preview

# 改变播放速度，仍然导出每一帧
python data/previews/PMC9883282/generate_preview.py --fps 10 --overwrite
```

可用 `--format mp4`、`--format webm` 或默认的 `--format both`。MP4 使用 MPEG-4 Part 2，适合本地播放器；WebM 使用 VP8，供浏览器播放。脚本复用已有 `h5py`、`numpy`、`opencv-python`，不需要 `ffmpeg` 命令或 Qt 窗口。如果当前 OpenCV 缺少某种编码器，会明确报错；可选择另一格式，或在已具备编码能力的本地环境运行，不必修改学校服务器的软件环境。

默认不覆盖任何已有预览产物。编码先写临时目录，所有所选视频通过帧数、尺寸和首帧解码检查后，才发布文件；编码失败会清理临时产物。`--overwrite` 只替换所选视频及本次的播放页／清单，不删除其他视频。若仅选择一段，本次生成的页面与清单也只列这一段；需要独立预览时建议另选输出目录。

输出包括：

- 每段的 `.mp4` 和／或 `.webm`；
- `index.html`：本地播放／下载入口；
- `manifest.json`：源文件、帧数、尺寸、预览速度与生成格式。

脚本不会重写本 README。原始 MAT、人工 mask、标注队列和训练缓存均不修改。若自定义输出到本仓库其他目录，需为该位置另加忽略规则；现有规则覆盖 `data/previews/` 下的预览产物。

## 3. 查看视频及帧号含义

可在 VS Code 资源管理器中右键下载视频，再在本地打开。浏览器观看时，将生成的 `index.html` 与 WebM 一起下载到同一目录后打开页面。

视频保留转置后的 **800×600 完整截图**，不裁剪、不叠加分割；左下角显示从 0 开始的原始帧号，对应标注图文件名末尾的帧号。逐帧读取 MAT，不把整段视频一次性载入内存。

**20 fps 仅为预览播放速度，不代表原始采集帧率。** 原始文件没有图像逐帧时间戳；截图上的 29 Hz 也不能证明导出帧的间隔。因此不配准、不叠加 force，视频时长也不能用于训练的时间标定。

## 4. 人工标注与已删除的 suggestions

180 张人工标注已完成，`suggestions/` 是早期模型预标注草稿，删除后不影响此脚本。当前训练读取 CSV 中已确认的 `mask_path`（指向 `masks/`），Web 标注也优先读取该人工 mask；CSV 中历史 `suggestion_path` 不会替代已确认的 mask。

无需为了预览重新运行 `prepare_pmc.py`，也无需重新生成 suggestions。标注导出工具仅在明确使用 checkpoint 且确有待预标注的训练帧时才创建 `suggestions/`，普通重跑不会创建空目录。

需要只读检查人工标注时，仍可运行：

```bash
python data/pipeline/dataPrepare.py --dataset pmc9883282 --validate-only
```
