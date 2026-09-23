# Data preparation and label conventions

**Start with the [dataset download guide](VIDEO_DATASETS_CN.md)** for public links, copyable commands, expected paths, storage requirements, and annotation setup. Mus-V is available from the project-provided [Google Drive file](https://drive.google.com/file/d/17dGwgo5UJsWUUGEENN9Zw9Tv3kurlya7/view?usp=drive_link).

Raw datasets, local human masks, checkpoints and generated artifacts are not included in a Git clone. In particular, the official PMC archive contains raw sequences, not this project's reviewed A/V masks. Exact reproduction requires the corresponding annotation snapshot and split metadata separately. Sample counts in historical reports refer to those experiment snapshots; current validity is recorded in each dataset's CSV.

[中文版](README_CN.md) · [Web annotation](collection/README_WEB_CN.md)

## 1. Dataset paths and training keys

Run commands from the repository root. Directory names differ from CLI/configuration keys:

| Directory under `data/datasets/` | Training key | `label_mode` | Training mask IDs | Supervision |
|---|---|---|---|---|
| `Mus-V/` | `musv` | `full3` | 0 / 1 / 2 | Background, vein, artery |
| `mendeley_data/` | `mendeley` | `artery` | 0 / 2 | Artery; other pixels do not distinguish vein from background |
| `PMC9883282/` | `pmc9883282` | `full3` | 0 / 1 / 2 | Reviewed A/V masks; source PNG 0 / 255 / 128 is remapped |
| `phantom_taobao/` | `phantom_taobao` | `vessel` | 0 / 3 | Untyped phantom vessel |
| `customer_3d_phantom/` | `customer_3d_phantom` | `vessel` | 0 / 3 | Untyped phantom vessel |

The model has three output channels. Label ID 3 means an untyped vessel and is supervised through the sum of vein and artery probabilities, not a fourth output. See [`pipeline/dataPrepare.py`](pipeline/dataPrepare.py) for the actual mappings.

ThrombUS, Regional-US and TUS-REC2024 are optional video sources, not registered training keys. Their original diagnostic, nerve or tracking annotations are not A/V segmentation masks. They still need format adapters and human A/V review.

Acquisition and annotation code lives in `collection/`, download/preprocessing code in `pipeline/`, and local datasets in the Git-ignored `datasets/`. Quality exclusions apply to both the source frame and its mask. Rebuild caches after annotation edits; do not treat unlabelled frames as empty ground truth.

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

The training entry point accepts `TRAIN_REPEATS=dataset:count,...` to override these
defaults per dataset instance. Length then becomes `n_originals × override_count`.
This does not modify global dataset configuration, cached arrays, labels or validation/
test data. Rebalancing requires no cache rebuild and creates no new human annotations.

| Dataset | `aug_times_train` | Why |
|---|---|---|
| `phantom_taobao` | 40 | counterweight, or the phantoms get diluted to ~7% of the mixed training set |
| `customer_3d_phantom` | 40 | same |
| `musv` | 8 | raised from 3 — the vein class was badly overfitting (train 0.855 / val 0.609) |
| `mendeley` | 5 | already 1100 frames |
| `pmc9883282` | 8 by default; overridden to 50 for this run | 120 training originals yield 6,000 rather than 960 draws; augmentation strength unchanged |

The original v11 run used this historical mix (not the current reviewed queue): **30,874 draws/epoch**: Mus-V 17,624 (57.1%),
Mendeley 3,850 (12.5%), commercial phantom 2,560 (8.3%), custom phantom 840 (2.7%),
and PMC 6,000 (19.4%). The phantoms have 64 and 21 training originals respectively.
PMC has 120 training originals, all from subject20; repeated augmentation adds neither
subjects nor independent observations. Validation and test retain 30 annotations each
from the two held-out subjects.

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
| `--dataset <name>` | which dataset to build: `phantom_taobao` / `customer_3d_phantom` / `mendeley` / `musv` / `pmc9883282` |
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
