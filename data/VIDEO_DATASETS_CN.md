# 数据集下载与准备指南

适用于首次 clone 本仓库的使用者。所有路径均相对于 `vein_segmentation/` 仓库根目录；原始数据、人工标签、缓存和视频不随 Git 分发。

**复现当前模型需要原有五个数据源；ThrombUS+、Regional-US、TUS-REC 2024 是可选扩展，不必全部下载。** 官方原始数据与本项目后续人工标注是两件事：尤其 PMC 官方包不包含本项目的人工 mask，下载后不能直接复现本地完整标注集。

## 1. Clone 与下载工具

```bash
git clone https://github.com/alfredzhang98/vein_segmentation.git
cd vein_segmentation
python -m pip install requests gdown
mkdir -p data/datasets
```

下载和解压本身不需要 GPU。进行预处理、标注或训练前，另按项目环境说明安装依赖：

```bash
python -m pip install -r requirements.txt
```

Linux/macOS 下检查存储：

```bash
df -h .
quota -s
```

`df` 显示文件系统余量；共享服务器还应以个人 `quota` 配额为准。没有启用配额的机器可能没有 `quota` 命令。解压文件、PNG、预标注和 NPZ 缓存会额外占空间，不能只按压缩包大小预留。

## 2. 数据集总览

| 数据集 | 公开下载入口 | 内容与现有标签 | 项目用途 / 目录 |
|---|---|---|---|
| **Mus-V** | [Google Drive：Mus-V.zip](https://drive.google.com/file/d/17dGwgo5UJsWUUGEENN9Zw9Tv3kurlya7/view?usp=drive_link) | 连续扫查帧，现成动静脉 mask | 原有训练域；`data/datasets/Mus-V/` |
| **Mendeley CCA** | [官方页面](https://data.mendeley.com/datasets/d4xt63mgjm/1) · [直接下载 ZIP](https://data.mendeley.com/public-api/zip/d4xt63mgjm/download/1) | 1,100 张超声图及动脉 mask；不是完整连续视频 | 原有训练域；`data/datasets/mendeley_data/` |
| **PMC9883282** | [原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883282/) · [直接下载 ZIP](https://pmc-oa-opendata.s3.amazonaws.com/PMC9883282.1/41598_2022_22867_MOESM1_ESM.zip) | 六段 MAT 原始序列，共 5,215 帧；**无官方分割 mask** | 原有训练域，需人工标注；`data/datasets/PMC9883282/` |
| **淘宝仿体** | [项目共享 Google Drive](https://drive.google.com/drive/folders/1LPwezlAUmxWnUo791RM3OPAhGQHq5r0v?usp=sharing) | 自采图像、血管 mask、metadata；不区分动静脉 | 原有训练域；`data/datasets/phantom_taobao/` |
| **自制 3D 仿体** | [同一共享目录](https://drive.google.com/drive/folders/1LPwezlAUmxWnUo791RM3OPAhGQHq5r0v?usp=sharing) | 自采图像、血管 mask、metadata；不区分动静脉 | 原有训练域；`data/datasets/customer_3d_phantom/` |
| **ThrombUS+** | [训练集](https://zenodo.org/records/17659415) · [测试集](https://zenodo.org/records/17664207) | 血管压缩 DICOM 视频；压缩性/血栓标签，**无 A/V 像素 mask** | 可选，优先研究压缩过程；`data/datasets/ThrombUS/` |
| **Regional-US** | [官方 GitHub](https://github.com/Regional-US/brachial_plexus) | 227 段超声视频；现成标签是神经/针，**不能当 A/V mask** | 可选，先筛选可见血管；`data/datasets/Regional-US/` |
| **TUS-REC 2024** | [官方主页及下载入口](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/) | 连续前臂扫查、位姿及地标；**无 A/V 分割 mask** | 可选，体积较大；`data/datasets/TUS-REC2024/` |

公共来源可以自行下载，不需要项目维护者代为传输。Google Drive 遇到临时下载限额时，可在浏览器打开上述链接下载；如果要求登录或没有权限，应检查共享权限，不要把登录页面保存成 ZIP。公共可访问不等于没有使用限制，各来源许可见第 7 节。

## 3. 原有数据集下载

### Mus-V

使用本项目提供的 [Google Drive 文件](https://drive.google.com/file/d/17dGwgo5UJsWUUGEENN9Zw9Tv3kurlya7/view?usp=drive_link)。也可直接在浏览器下载 `Mus-V.zip`。

```bash
gdown 17dGwgo5UJsWUUGEENN9Zw9Tv3kurlya7 -O data/datasets/Mus-V.zip
python -m zipfile -l data/datasets/Mus-V.zip
```

先看压缩包顶层：如果顶层为 `Mus-V/`，解压到 `data/datasets/`；如果顶层直接为 `Multimodal Ultrasound Vascular Image Segmentation/`，解压到 `data/datasets/Mus-V/`。两种命令只执行符合压缩包结构的一条：

```bash
# ZIP 顶层为 Mus-V/ 时
python -m zipfile -e data/datasets/Mus-V.zip data/datasets/

# ZIP 顶层直接为 Multimodal Ultrasound Vascular Image Segmentation/ 时
python -m zipfile -e data/datasets/Mus-V.zip data/datasets/Mus-V/
```

最终必须匹配：

```text
data/datasets/Mus-V/
└── Multimodal Ultrasound Vascular Image Segmentation/
    ├── Videos/
    │   ├── train/<sequence>/*.png
    │   └── valid/<sequence>/*.png
    └── Annotations/<sequence>/*.png
```

不要出现 `Mus-V/Mus-V/` 双层目录。保持序列名和帧号不变；mask 原始值为背景 0、静脉 1、动脉 2。

### Mendeley CCA

官方页面的 **Download All** 与下面的公开下载地址对应同一版本：

```bash
mkdir -p data/datasets/mendeley_data
curl -fL --retry 3 \
  'https://data.mendeley.com/public-api/zip/d4xt63mgjm/download/1' \
  -o data/datasets/mendeley_data/cca.zip
python -m zipfile -e data/datasets/mendeley_data/cca.zip data/datasets/mendeley_data/
```

核对并整理为以下结构；如果下载包直接给出两个子目录，将它们放入 `Common Carotid Artery Ultrasound Images/`：

```text
data/datasets/mendeley_data/Common Carotid Artery Ultrasound Images/
├── US images/*.png
└── Expert mask images/*.png
```

这里只有动脉标签。预处理使用 `label_mode=artery`，通过 partition CE＋Dice 监督，不能把未标注的静脉当作已知背景。

### PMC9883282 原始视频

原包约 **0.64 GB**，解压约 **0.64 GB**。以下用于首次准备；已有人工标注时，不要覆盖 `images/`、`masks/` 或 CSV。

```bash
mkdir -p data/datasets/PMC9883282
curl -fL --retry 3 \
  'https://pmc-oa-opendata.s3.amazonaws.com/PMC9883282.1/41598_2022_22867_MOESM1_ESM.zip' \
  -o data/datasets/PMC9883282/41598_2022_22867_MOESM1_ESM.zip
python -m zipfile -e \
  data/datasets/PMC9883282/41598_2022_22867_MOESM1_ESM.zip \
  data/datasets/PMC9883282/
mv -n data/datasets/PMC9883282/cvp_ijv_fcu_rawdata/*.mat data/datasets/PMC9883282/
mv -n data/datasets/PMC9883282/cvp_ijv_fcu_rawdata/readme.txt data/datasets/PMC9883282/
python data/previews/PMC9883282/generate_preview.py --check-only
```

检查通过后，**首次**建立标注队列并启动网页：

```bash
python data/pipeline/prepare_pmc.py --frames-per-sequence 30
python data/collection/label_web.py PMC9883282 1 --port 8765
```

这会建立 180 张待标注队列，不会自动获得本项目后来扩充到 362 张的人工标签。可加 `--checkpoint models/unet/checkpoints/unet_v11.pth` 生成训练帧的辅助轮廓，但须先自行提供该权重；checkpoint 不随 Git clone 分发。模型草稿必须人工确认，验证/测试帧不加载模型建议。

**需要精确复现本地 PMC 结果时，还需同一版人工 masks、metadata CSV、有效性标记和数据划分。** 目前没有在本指南中提供该人工标注快照的公开下载地址，不能将官方原包说成已经包含这些标签。

### 两份仿体

打开 [共享目录](https://drive.google.com/drive/folders/1LPwezlAUmxWnUo791RM3OPAhGQHq5r0v?usp=sharing)，下载两份仿体资料；如希望命令行下载目录，可用：

```bash
gdown --folder 'https://drive.google.com/drive/folders/1LPwezlAUmxWnUo791RM3OPAhGQHq5r0v' \
  -O data/datasets/phantom_download/
```

按原包格式解压，将旧名 `phantom_1` 对应的图像、标签和 CSV 整理到 `phantom_taobao/`，将 `phantom_2` 整理到 `customer_3d_phantom/`。最终要求：

```text
data/datasets/
├── phantom_taobao/
│   ├── images/
│   ├── masks/
│   └── meta_phantom_taobao_1.csv
└── customer_3d_phantom/
    ├── images/
    ├── masks/
    └── meta_customer_3d_phantom_1.csv
```

若原包 CSV 仍使用旧名，也需对应改名。CSV 的 `relative_path`、`mask_path` 如果保留了原作者机器上的路径，整理好上述目录后运行下面的路径更新。它先检查所有图像/标签存在，再写回 CSV；保留其他字段与未标注状态：

```bash
python - <<'PY'
import csv, io
from pathlib import Path
for name in ('phantom_taobao', 'customer_3d_phantom'):
    root = Path('data/datasets') / name
    meta = root / f'meta_{name}_1.csv'
    with meta.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        image = root / 'images' / Path(row['filename'].replace('\\', '/')).name
        assert image.is_file(), image
        row['relative_path'] = image.as_posix()
        old_mask = row.get('mask_path', '').strip()
        if old_mask and old_mask.lower() not in ('nan', 'none'):
            mask = root / 'masks' / Path(old_mask.replace('\\', '/')).name
            assert mask.is_file(), mask
            row['mask_path'] = mask.as_posix()
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    temporary = meta.with_suffix('.csv.tmp')
    temporary.write_text(text.getvalue(), encoding='utf-8')
    temporary.replace(meta)
PY
```

两份仿体是未分型血管标签，训练时用 `label_mode=vessel`。

## 4. 三个可选视频数据集

下载脚本只依赖 `requests` 和 Python 标准库，支持续传、校验、解压与状态记录：Zenodo 使用官方 MD5，Regional-US 使用本项目记录的固定提交 ZIP 的 SHA-256。**必须显式指定数据集，没有“默认全下载”。** `--dry-run` 只查询元数据、列出文件与大小，不下载数据包，不创建数据目录。

```bash
# 先查看所选来源、文件名与压缩包大小
python -m data.pipeline.download_video_datasets --datasets Regional-US --dry-run

# 以下按需选择执行，不必全部运行
python -m data.pipeline.download_video_datasets --datasets Regional-US
python -m data.pipeline.download_video_datasets --datasets ThrombUS
python -m data.pipeline.download_video_datasets --datasets TUS-REC2024
```

也支持只下载某个原包，例如 ThrombUS 训练集或 TUS-REC 验证集：

```bash
python -m data.pipeline.download_video_datasets --datasets ThrombUS \
  --files Thrombus_ultrasound_dataset_1_training.zip
python -m data.pipeline.download_video_datasets --datasets TUS-REC2024 \
  --files Freehand_US_data_val.zip --dry-run
# 确认空间足够后，去掉 --dry-run 才会下载与解压。
```

不要将官方验证集改作训练集后，又用它报告验证成绩。Regional-US 固定到已核查的提交 `ac4bf778db01cc81338773485e59a8ac254f665c`，保证文档和下载内容一致。

统一输出结构，不再按某轮实验创建另一份数据集目录：

```text
data/datasets/<数据集>/
├── archives/                 # 原始 ZIP；下载中有 .part / .chunks
├── raw/<压缩包名称>/         # 解压后的原始目录，保留官方结构
├── download_sources.json    # 公开下载地址、大小与校验和
└── download_status.json     # 下载、校验、解压状态
```

重复同一命令可继续未完成的下载。显示 `complete` 表示对应原包已校验并解压，不代表动静脉标注已经完成。已解压目录中的 `.extracted.json` 用于识别完成状态；不要单独删除原文件而保留这个标记。

### 空间预算

以下为 2026-09-23 核查的十进制 GB；不含 PNG、人工标注、训练缓存及文件系统开销。分块下载在合并时还可能暂时保留一份额外的压缩数据。

| 数据集 | 下载压缩包 | 解压后 | 压缩包＋解压文件 |
|---|---:|---:|---:|
| ThrombUS+ | 12.19 | 13.21 | 25.40 |
| Regional-US | 0.65 | 0.84 | 1.49 |
| TUS-REC 2024 | 88.60 | 197.06 | 285.67 |
| 合计 | 101.44 | 211.11 | 312.56 |

ThrombUS 训练/测试包分别为 9.73/2.46 GB。TUS-REC 训练 Part 1/2 分别为 43.38/40.38 GB，验证集 4.84 GB；Part 3 为约 185 KB 的地标文件，另有标定 CSV。TUS-REC 验证集单独解压也约需 10.68 GB。

## 5. 标注与时序数据准备

PMC 网页启动后，打开终端打印的 **带 `#token=...` 的完整链接**。远程服务器通过 VS Code Ports 转发 8765，或使用 `ssh -N -L 8765:127.0.0.1:8765 用户@服务器`。完整操作见 [网页标注说明](collection/README_WEB_CN.md)。

三份可选数据的下载脚本**只负责原始数据**，尚未提供自动 DICOM/HDF5/MP4 转换并建立 A/V 标注队列的完整入口；不能仅下载后就运行 PMC 的导出器。接入网页前，需要按原始受试者、序列、帧号导出 PNG，创建对应 `meta_<数据集>_1.csv`，把原有非 A/V 标签保留在 `raw/`，新 A/V 标签状态设为 `ND`。准备好后可复用 `label_web.py <数据集名> 1 --port 8765`；每次选择一个已准备好的数据集。

标注时保留这些规则：

- 原始时间顺序、帧号与受试者身份不能丢失；同一人的相邻片段不能分到训练和测试两侧。
- 无接触、无有效组织回声的帧排除；组织正常而血管压扁、消失的帧保留。无效帧切断时序片段。
- 神经 mask、血栓分类、探头位姿都不能充当动静脉像素标签；未标注帧不能用空 mask 代替。
- v11 只提供辅助轮廓，人工确认才成为真值。PMC 保存时的“每类一个主要连通区域”规则不自动应用到可能有多条血管的新数据集。
- ThrombUS 优先检查 CFV 动静脉切面；Regional-US 需筛选血管可见的片段；TUS-REC 是探头扫过不同切面，不能把它直接等同于固定切面压缩。

## 6. 下载后生成训练缓存

只对已经下载、路径正确且标签可用的数据集执行对应命令：

```bash
python data/pipeline/dataPrepare.py --dataset musv --no-png
python data/pipeline/dataPrepare.py --dataset mendeley --no-png
python data/pipeline/dataPrepare.py --dataset phantom_taobao --no-png
python data/pipeline/dataPrepare.py --dataset customer_3d_phantom --no-png

# PMC 需要先完成标注复核
python data/pipeline/dataPrepare.py --dataset pmc9883282 --validate-only
python data/pipeline/dataPrepare.py --dataset pmc9883282 --no-png
```

ThrombUS、Regional-US、TUS-REC2024 目前不在 `DATASET_CONFIGS` 的训练入口中，完成适配与标注前不要直接填入 `DATASET`。数据处理、标签映射和 partition loss 说明见 [数据主文档](README_CN.md)。

## 7. 使用条款与可复现范围

| 来源 | 使用说明 |
|---|---|
| Mus-V | 提供下载镜像；原来源为 [Kaggle](https://www.kaggle.com/datasets/fa8b3e1386722702d9c80a7d2d10d5d50eef20d14a604078b38d01c66fd9f356)，沿用来源的 CC BY-NC 4.0，镜像不改变许可 |
| Mendeley CCA / PMC 原始资料 | 官方页面标示 CC BY 4.0；引用对应数据与论文 |
| 两份仿体 | 本项目自采共享数据，按共享资料中的说明使用；不要把代码许可自动当作数据许可 |
| ThrombUS+ | Zenodo 所列 CC BY 4.0；保留训练/测试来源信息 |
| Regional-US | 原仓库声明非商业数据使用协议 |
| TUS-REC 2024 | 官方允许研究使用并要求引用，禁止商业使用；2025 数据的规则不同，不能混用 |

本仓库上传源码、配置与文档。`data/datasets/`、缓存、权重、`runs/`、`generated/` 和本地测试均不上传。新 clone 不会自带本机绝对路径、人工标注快照、模型 checkpoint 或已经生成的视频；每项产物需按其说明获取或重建。
