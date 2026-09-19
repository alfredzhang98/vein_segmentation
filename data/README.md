# Data

Everything to do with data lives under this directory: how frames are acquired, how they
become training tensors, where the datasets go, and the vendor SDK the acquisition
scripts need.

> 中文版：[README_CN.md](README_CN.md)

## Layout

| Directory | What it holds | Tracked by git |
|---|---|---|
| [`collection/`](collection/) | Acquisition and annotation — Clarius Cast, A325 capture card, the brush labeller | yes |
| [`pipeline/`](pipeline/) | Preprocessing, augmentation, the NPZ cache, the `Dataset` class | yes |
| [`datasets/`](datasets/) | The actual data. Download it yourself — see §1 | **no** |
| [`clarius_sdk/`](clarius_sdk/) | Clarius Cast native API. Needed **only** for acquisition, never for training | yes |
| [`samples/`](samples/) | A handful of example frames, so the demo commands in the README run out of the box | yes |

Run every command from the **repository root**. Paths in `pipeline/dataPrepare.py` are
relative (`data/datasets/...`).

---

## 1. Datasets

Nothing under `datasets/` is tracked. Two of the four sets are under licences that do not
permit redistribution here, and the other two are large. Download each one and place it
at the exact path below.

Directory names matter: they are the keys in `DATASET_CONFIGS`
([`pipeline/dataPrepare.py`](pipeline/dataPrepare.py)) and the values accepted by
`DATASET` / `VAL_DATASET` in [`.env`](../.env).

| Directory | Source | `label_mode` | Mask ids | What is annotated |
|---|---|---|---|---|
| `Mus-V/` | Kaggle (public) | `full3` | 0 / 1 / 2 | background, vein, artery — all three |
| `mendeley_data/` | Mendeley Data (public) | `artery` | 0 / 2 | common carotid artery only |
| `phantom_taobao/` | self-collected | `vessel` | 0 / 3 | a gel tube, type undetermined |
| `customer_3d_phantom/` | self-collected | `vessel` | 0 / 3 | a gel tube, type undetermined |

Mask id `3` means "a vessel, type not labelled" and never appears as a model output
channel — the model emits 3 channels (background / vein / artery). The mapping from
whatever a dataset stores on disk to these shared ids is `mask_class_map`, declared per
dataset in `DATASET_CONFIGS`.

### 1.1 `phantom_taobao/` and `customer_3d_phantom/` — self-collected

Download from
[Google Drive](https://drive.google.com/drive/folders/1LPwezlAUmxWnUo791RM3OPAhGQHq5r0v?usp=sharing)
and extract **directly into `datasets/`** — do not create a nested `datasets/data/...`.

> **Rename on extraction.** The archives use the older names. `phantom_1` is the
> commercial phantom and must become **`phantom_taobao`**; `phantom_2` is the custom
> phantom and must become **`customer_3d_phantom`**. The code will not find them under
> the old names.

```
data/datasets/
├── phantom_taobao/                      # commercial phantom, Clarius Cast capture
│   ├── images/                          #   raw grayscale frames
│   ├── masks/                           #   annotated binary masks
│   ├── images_aug/  masks_aug/          #   optional PNG previews from dataPrepare.py
│   └── meta_phantom_taobao_1.csv        #   per-image metadata (micropixel, depth, gain)
│
└── customer_3d_phantom/                 # custom gelatin + agar + saline phantom, A325 capture
    ├── images/  masks/
    ├── images_aug/  masks_aug/
    └── meta_customer_3d_phantom_1.csv
```

```bash
python data/pipeline/dataPrepare.py --dataset phantom_taobao
python data/pipeline/dataPrepare.py --dataset customer_3d_phantom
```

- `phantom_taobao` — commercial phantom, [Taobao listing](https://item.taobao.com/item.htm?id=762322402710)
- `customer_3d_phantom` — custom-made (gelatin + agar + saline), 3D-printed vessel core

### 1.2 `mendeley_data/` — Common Carotid Artery Ultrasound Images

**Not included.** Download from
[Mendeley Data](https://data.mendeley.com/datasets/d4xt63mgjm/1) (CC BY 4.0) and extract
so the paths are exactly:

```
data/datasets/mendeley_data/
└── Common Carotid Artery Ultrasound Images/
    ├── US images/              # 1100 frames (PNG, 709x749x3)
    └── Expert mask images/     # 1100 expert masks, matching filenames
```

```bash
python data/pipeline/dataPrepare.py --dataset mendeley --no-png
```

**Specs** — 1100 images + 1100 expert masks, 709×749×3; Mindray UMT-500Plus with an
L13-3s linear probe; 11 subjects; CC BY 4.0.

**Why `label_mode = artery`.** The expert masks label the common carotid artery and
nothing else. The internal jugular vein is frequently in frame but left unlabelled,
sitting inside the "background" region. Supervising those pixels as background would
teach the model that veins look like background. The objective therefore constrains the
artery marginal `p₂` only, and is invariant to how the rest of the probability splits
between background and vein — so an unlabelled jugular vein predicted here costs exactly
nothing.

> Where the loss is documented: the **mathematics** is in the [top-level README](../README.md);
> **which partition each dataset induces** is the `label_mode` column above; the
> **implementation** is in [`models/unet/README.md`](../models/unet/README.md#the-loss--partial_label_loss).

### 1.3 `Mus-V/` — Carotid and Femoral Vessel Ultrasound Dataset

**Not included.** Download from
[Kaggle](https://www.kaggle.com/datasets/fa8b3e1386722702d9c80a7d2d10d5d50eef20d14a604078b38d01c66fd9f356)
(CC BY-NC 4.0, **non-commercial only**) and extract so the paths are:

```
data/datasets/Mus-V/
└── Multimodal Ultrasound Vascular Image Segmentation/
    └── ...                     # as distributed
```

```bash
python data/pipeline/dataPrepare.py --dataset musv
```

**Specs** — 3114 frames (2203 train + 911 validation) from 105 probe sweeps, 11
volunteers; Angell Pioneer H20 scanner; CC BY-NC 4.0.

**Class identity.** Masks are already `{0, 1, 2}` label maps, so `mask_class_map` is the
identity. That class 1 is the **vein** and class 2 the **artery** was first inferred from
area statistics — class 1 collapses to zero area in 272 frames and has 3.6× the area
variance of class 2, i.e. it is the compressible one — and is independently confirmed by
Mendeley: the channel trained as class 2 from Mus-V predicts the Mendeley expert *artery*
masks at Dice 0.956. Were the two swapped that number would be near zero.

This is the only dataset that annotates vein and artery separately, and therefore the
only source of the vein/artery prior that the phantom and CCA sets inherit.

### 1.4 `PMC9883282/` — IJV force-collapse recordings (optional, **unlabelled**)

Not used for training. Supplementary data from a Scientific Reports study of internal
jugular vein collapse under probe force: 6 MATLAB v7.3 files, each holding
`ultrasound_images (N, 800, 600) uint8`, `force_data` and `time_data`, for 3 of the
study's 27 subjects.

**It contains no masks of any kind.** It cannot enter the training objective — a dataset
has to determine at least one group of the partition to supervise anything, and this one
determines none. It is kept only as a reference for the vein-compression failure mode
discussed in [`results/unet/README.md`](../results/unet/README.md), and
would need cropping (the frames are full scanner screenshots with the UI overlay) plus
resampling (force is ~50 Hz against 29 Hz video) before any use.

```bash
curl -O https://pmc-oa-opendata.s3.amazonaws.com/PMC9883282.1/41598_2022_22867_MOESM1_ESM.zip
```

---

## 2. From frames to tensors

[`pipeline/dataPrepare.py`](pipeline/dataPrepare.py) turns a downloaded dataset into an
NPZ cache, and [`ReadDataset`](pipeline/dataPrepare.py) turns that cache into batches.

### 2.1 Split first, then augment

The train/val/test split is decided on **original frames**, before any augmentation, with
a fixed `seed=42`. Only the train split is ever augmented; val and test are resized and
normalised and nothing else, so the metrics mean what they say.

```
original frames (per dataset)
  ├─ train  70%  → augmented online, fresh every epoch
  ├─ val    15%  → resize + normalise only
  └─ test   15%  → resize + normalise only
```

Mus-V is split **by probe sweep**, not by frame. Consecutive frames of one sweep are near
duplicates; a frame-level split would put the same vessel in both train and test.

### 2.2 The NPZ cache holds originals, not augmented copies

This is the part most people get wrong. The cache stores the **resized original frames**:

| Field | Shape | dtype | Meaning |
|---|---|---|---|
| `images` | `(N, 576, 544)` | `uint8` | grayscale frames, already resized and cropped |
| `masks` | `(N, 576, 544)` | `uint8` | class ids in the shared space (0/1/2/3) |
| `image_type` | `(N,)` | `str` | provenance tag |
| `metadata` | `(1,)` | `dict` | the augmentation config fingerprint |

Augmentation runs **online, in `__getitem__`**. `ReadDataset.__len__` returns
`n_originals × aug_times_train`, and every draw of the same original gets an independent
random augmentation — a different one again next epoch. So `aug_times_train` sets how
many *fresh* samples per epoch each original produces, not how many fixed copies sit on
disk.

| Dataset | `aug_times_train` | Why |
|---|---|---|
| `phantom_taobao` | 40 | counterweight, or the phantoms get diluted to ~7% of the mixed training set |
| `customer_3d_phantom` | 40 | same |
| `musv` | 8 | raised from 3 — the vein class was badly overfitting (train 0.855 / val 0.609) |
| `mendeley` | 5 | already 1100 frames |

### 2.3 The augmentation pipeline

Applied to train samples only, in this order. Geometry acts on image **and** mask
together; appearance acts on the image only.

| # | Step | Probability | Image | Mask |
|---|---|---|---|---|
| 1 | `isotropic_zoom` — one uniform scale factor | if `aug_scale` set | ✅ | ✅ |
| 2 | Resize to 1.5× target, `INTER_CUBIC` | always | ✅ | ✅ |
| 3 | Rotate ±10°, `BORDER_REPLICATE` | 0.5 | ✅ | ✅ |
| 4 | Horizontal flip | 0.5 | ✅ | ✅ |
| 5 | Vertical flip | 0.5 | ✅ | ✅ |
| 6 | Resize to 576×544, `INTER_AREA` | always | ✅ | ✅ |
| 7 | Brightness / contrast ±10% | 0.6 (0.7 for phantoms) | ✅ | ❌ |
| 8 | Gaussian noise, σ ∈ [0.05, 0.1] | 0.3 | ✅ | ❌ |
| 9 | Random gamma | 0.5, if `aug_gamma` set | ✅ | ❌ |
| 10 | Multiplicative (speckle) noise | 0.4, if `aug_speckle` set | ✅ | ❌ |
| 11 | Gaussian blur **or** sharpen | if `aug_blur` set | ✅ | ❌ |
| 12 | Normalise → tensor | always | ✅ | ❌ |

Two choices here are deliberate and worth keeping:

**Upscale-then-downscale.** Step 2 enlarges to 1.5× *before* rotation so the rotation has
real pixels to pull from at the corners, then step 6 brings it back down. Rotating at
target size leaves black wedges.

**The zoom is hand-rolled, not `A.Affine(scale=…)`.** Albumentations samples the x and y
scale independently, which turns a circle into anything from a 0.70:1 to a 1.47:1
ellipse. Artery-versus-vein is a **shape** judgement — an artery is round, a vein
collapses — so an anisotropic zoom would corrupt the one cue the model has.

After step 12: image is `(1, 576, 544)` float32 in `[-1, 1]` via `(x/255 − 0.5) / 0.5`;
mask is `(1, 576, 544)` int64 class ids, untouched by every appearance step.

### 2.4 Cache invalidation

`dataPrepare.py` stores a fingerprint of the augmentation config next to the cache. Change
an augmentation parameter and the cache is stale:

```bash
python data/pipeline/dataPrepare.py --check-stale       # report only, changes nothing
python data/pipeline/dataPrepare.py --dataset <name>    # rebuild
```

| Flag | Effect |
|---|---|
| `--dataset <name>` | which dataset to build: `phantom_taobao` / `customer_3d_phantom` / `mendeley` / `musv` |
| `--no-png` | skip writing PNG previews (the NPZ is still built) — much faster on the large sets |
| `--test-loader` | after building, instantiate the DataLoader and print batch shapes |
| `--check-stale` | report which caches no longer match their config |

### 2.5 Adding a dataset

Add an entry to `DATASET_CONFIGS` in [`pipeline/dataPrepare.py`](pipeline/dataPrepare.py):

```python
"my_dataset": {
    "source_type":     "folder",                       # "csv" | "folder" | "musv"
    "data_dir":        Path("data/datasets/my_dataset"),
    "images_dir":      Path("data/datasets/my_dataset/images"),
    "masks_dir":       Path("data/datasets/my_dataset/masks"),
    "npz_out_dir":     Path("data/datasets/my_dataset"),
    "npz_prefix":      "augmented_my_dataset_1",
    "aug_times_train": 10,                             # scale to the dataset's size
    "mask_class_map":  {0: 0, 255: 3},                 # on-disk value -> shared class id
    "label_mode":      "vessel",                       # full3 | artery | vessel
    "target_size":     (576, 544),
    "color_mode":      "gray",                         # "gray" | "rgb_to_gray"
    "crop":            None,                           # None | (top, bottom, left, right)
    "seed":            42,
},
```

`label_mode` is the important one — it declares which groups of the class partition this
dataset's masks actually determine, and therefore what the loss is allowed to supervise.
See the loss section of the [top-level README](../README.md).

Then build it:

```bash
python data/pipeline/dataPrepare.py --dataset my_dataset --no-png
```

---

## 3. Collection

[`collection/`](collection/) is the acquisition side. It is independent of training — you
only need it if you are capturing new data.

| File | What it does |
|---|---|
| [`collect_clarius.py`](collection/collect_clarius.py) | Capture from a Clarius probe over the Cast API |
| [`collect_a325.py`](collection/collect_a325.py) | Capture from an A325 capture card |
| [`label.py`](collection/label.py) | Circular-brush annotation tool (PySide6) |
| [`config.py`](collection/config.py) | Collection-side paths and the mask value `label.py` paints |

`config.py` and `pipeline/dataPrepare.py` have to agree on mask values: a value painted by
`label.py` that no `mask_class_map` mentions is silently dropped at training time.

---

## 4. Clarius SDK

[`clarius_sdk/`](clarius_sdk/) holds the Clarius Cast native API — shared libraries, C
headers, the upstream `cast` submodule and Clarius's own example programs. It is needed
**only** for native-API development and acquisition; training and inference never touch
it. See [`clarius_sdk/README.md`](clarius_sdk/README.md).
