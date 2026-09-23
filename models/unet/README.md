# U-Net

> The first S1 experiment is available in [models/temporal](../../results/temporal/RESULTS_CN.md) with [paired results](../../results/temporal/RESULTS_CN.md). Temporal benefit has not passed acceptance; v11 remains the reference. The unchanged mixed loss is shared through `models/losses.py`.


> Cross-model temporal perception and world-model design: [shared development roadmap](../../results/temporal/RESULTS_CN.md). This directory documents the U-Net implementation, training and inference.

> 中文版：[README_CN.md](README_CN.md)

The segmentation model: a U-Net emitting **three channels — background, vein, artery**,
trained with the partial-label objective described in the
[top-level README](../../README.md).

Everything this architecture needs lives in this folder. Adding another architecture
means adding `models/<arch>/` next to it with the same five files; nothing under `data/`
has to change.

| File | What it is |
|---|---|
| [`model.py`](model.py) | The network. `UNet(n_channels, n_classes, bilinear, base_ch)` |
| [`parts.py`](parts.py) | Its building blocks (double conv, down, up, out) |
| [`train.py`](train.py) | Training: the partial-label loss, validation, checkpointing |
| [`test.py`](evaluate.py) | Held-out evaluation, post-processing, FPS |
| [`infer.py`](infer.py) | **Deployment inference.** Self-contained — only torch / numpy / cv2 + `model.py` |
| `checkpoints/` | Weights. Not tracked by git; `unet_v10.pth` is the released one |

**Released checkpoint**: `models/unet/checkpoints/unet_v10.pth` — `base_ch=32`,
7 762 531 parameters, dropout2d 0.3, epoch 19, best PRIMARY 0.8622. Results and the full
experiment log are in [`results/unet/README.md`](../../results/unet/README.md).

**Run every command from the repository root.**

---

## The loss — `partial_label_loss()`

The mathematics and the reason the objective exists are in the
[top-level README](../../README.md); which partition each dataset induces is in
[`data/README.md`](../../data/README.md). This section is the **implementation**, which
lives in [`train.py`](train.py).

```python
partial_label_loss(logits,          # (B, 3, H, W) raw
                   target,          # (B, 1, H, W) long, ids in that sample's own label space
                   modes,           # list of len B: "full3" | "vessel" | "artery" per sample
                   class_weights=None,
                   cfg=None)
```

`log_softmax` is taken **once** over the whole batch, then the batch is partitioned by
`label_mode` and each group gets its own branch. A batch may freely mix all three modes —
that is the normal case when `DATASET` joins four sets with `+`.

| `label_mode` | Foreground log-probability | Background log-probability | Ids allowed in the target |
|---|---|---|---|
| `full3` | — (3-way `nll_loss` over all channels) | — | `{0 bg, 1 vein, 2 artery}` |
| `vessel` | `logsumexp(lp[:, [1, 2]])` = `log(p₁+p₂)` | `lp[:, 0]` = `log p₀` | `{0 bg, 3 vessel-untyped}` |
| `artery` | `lp[:, 2]` = `log p₂` | `logsumexp(lp[:, [0, 1]])` = `log(p₀+p₁)` | `{0 bg, 2 artery}` |

Four implementation points that are not obvious from the formula:

**1. The ids are the meaning, and a mis-route is asserted, not tolerated.** Each branch
asserts the target contains only the ids that mode is allowed to carry (`MODE_IDS` in
[`data/pipeline/dataPrepare.py`](../../data/pipeline/dataPrepare.py)). A sample routed to
the wrong marginal would supervise the wrong channel and merely look like a bad epoch —
it would never raise. The assert turns a silent corruption into a crash.

**2. `full3`'s region term averages the vessel classes only, not all three.** A 3-class
Dice average is dominated by the background channel: it is ~96% of pixels and always
near-perfect, so including it drowns out the vein term the loss exists to move. The code
averages `_region_loss` over `CLASS_VEIN` and `CLASS_ARTERY` alone.

**3. Everything stays in the log domain.** `log(p₁+p₂)` is computed with `logsumexp`, not
`log(exp+exp)`, so it is exact and safe under AMP. For the two-group modes,
`log_bg = log(1 − p_fg)` is likewise read off a real channel rather than reconstructed by
subtraction.

**4. The modes are combined by sample count, not by group count.** Each branch's loss is
multiplied by how many samples of that mode are in the batch, summed, and divided by the
batch size — a sample-count-weighted mean. A batch that happens to hold 15 Mus-V frames
and 1 phantom frame is weighted accordingly.

### The region term: Dice, or Tversky

`_region_loss` is `1 − Dice` by default. Setting `TVERSKY_BETA` in `.env` switches it to
`1 − Tversky`, where `alpha` prices false positives and `beta` prices false negatives
(`alpha = beta = 0.5` reduces exactly to Dice).

It exists because of a measured asymmetry: on Mus-V val, **37% of true vein pixels were
being called background while only 1.3% were called artery** — the model was not
confusing the two vessels, it was missing the vein outright. A vein is thin, compressed
against the tissue and low-contrast, so under a loss that prices a false positive and a
false negative identically, the safe play is to predict background. `beta > alpha` makes
missing a vein pixel more expensive than inventing one, which is the asymmetry the
problem actually has. **It is off in the shipped `.env`** — the released checkpoint uses
plain Dice.

> `CE_CLASS_WEIGHTS` is deliberately left off too. On the missed vein pixels the median
> `p₁` is 0.0019 against `p₀` = 0.946; re-weighting the cross-entropy does not reach a
> probability that small.

---

## train.py — training

`train.py` reads the repository [`.env`](../../.env) by default. Use
`--config configs/train.env` for the generic four-domain, from-scratch template.
An explicit config replaces the root `.env`; shell environment variables take
precedence. Data and checkpoint paths are repository-relative.

Use `--set KEY=VALUE [KEY=VALUE ...]` to override datasets, run names, weights
and hyperparameters without creating another config file. Precedence is CLI overrides,
shell variables, the selected file, then code defaults. `--show-config` prints the
effective settings and training mode without loading data or selecting a GPU. `INIT_PATH` loads model weights only,
whereas `RESUME_PATH` restores training state; they are mutually exclusive.
For `reviewed_csv` datasets, validate with
`python data/pipeline/dataPrepare.py --dataset pmc9883282 --validate-only`,
then build caches with `--no-png` instead of `--validate-only` before training.

`TRAIN_REPEATS=pmc9883282:50` overrides online draws per training original for the
selected source. Use commas for multiple overrides; unspecified sources retain
`aug_times_train`. Values must be positive integers and sources must be selected in
`DATASET`. This changes sampling frequency, not augmentation strength, label semantics,
losses or held-out splits. No NPZ rebuild is required. Checkpoints record both the
requested overrides and the effective per-source repeats.

The current five-domain rebalanced run uses 120×50 = 6,000 PMC draws out of 30,874 per
epoch (19.4%), with the other four domains unchanged. The completed `unet_v11_balanced` run is archived as `checkpoints/unet_v11.pth`,
beside v10. Its abandoned predecessor and empty version subdirectories were removed.
`VAL_WEIGHTS` controls checkpoint selection, independently of training exposure.

```bash
python models/unet/train.py
```

It picks an idle GPU by itself (see [`gpu_utils.py`](../../gpu_utils.py)), builds the
dataloaders from the NPZ caches, and writes checkpoints to `CHECKPOINT_DIR`.

The settings you will actually touch:

| `.env` key | Default | What it does |
|---|---|---|
| `DATASET` | `musv+phantom_taobao+mendeley+customer_3d_phantom` | Training set. Join with `+` to train on the union — each sub-dataset carries its own `label_mode`, and the loss routes every sample to the right supervision mode |
| `TRAIN_REPEATS` | empty; source defaults | Online draws per training original, e.g. `pmc9883282:50`; no effect on validation/test |
| `VAL_DATASET` | `musv,phantom_taobao+customer_3d_phantom,mendeley` | Validation sets, comma-separated. `+` pools two into one metric |
| `VAL_WEIGHTS` | `phantom…:0.45, musv:0.45, mendeley:0.10` | Deployment priority, used to combine validation scores into the checkpoint criterion |
| `NUM_CLASSES` | `3` | 3 = background / vein / artery. 1 = the legacy binary model |
| `BASE_CH` | `32` | Encoder width. 32 → 7.8 M parameters; 64 → 31 M |
| `DROPOUT` | `0.3` | Dropout2d on the two deepest encoder blocks and the bottleneck |
| `LEARNING_RATE` | `3e-4` | |
| `BATCH_SIZE_PER_GPU` | `16` | |
| `EPOCHS` / `EARLY_STOP_PATIENCE` | `200` / `30` | Patience must be well above `PLATEAU_PATIENCE`, or the run stops before the LR drop can take effect |
| `RUN_NAME` | `mixav_v3_cap32` | Goes into the checkpoint filename and the W&B run name. **Change it before every experiment**, or filenames collide |
| `INIT_PATH` | empty | Fine-tune from model weights only |
| `RESUME_PATH` | empty | Resume model, optimizer, scheduler and epoch; mutually exclusive with INIT_PATH |
| `FREEZE_ENCODER` | `false` | Freeze the encoder; `UNFREEZE_EPOCH` / `UNFREEZE_LAYERS` control the thaw |
| `CHECKPOINT_DIR` | `models/unet/checkpoints` | Where weights are written |
| `MAX_GPUS` | `1` | Keep at 1. Multi-GPU DataParallel is the only configuration ever seen to turn val loss into NaN |
| `WANDB_ENABLE` | `true` | Set `false` to train without experiment logging |

Settings can also be overridden per run from the shell, without editing `.env`:

```bash
RUN_NAME=expG_musv_3class DATASET=musv VAL_DATASET=musv \
python models/unet/train.py
```

**Checkpoint selection** (`primary_score` in `train.py`): if the primary validation set
is fully annotated (Mus-V), the score is `(dice_vein + dice_artery)/2` — separating the
two *is* the task, and a merged `dice_vessel` stays high even when the model confuses
them. Otherwise it falls back to `dice_vessel`.

---

## test.py — evaluation

```bash
python models/unet/evaluate.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --dataset musv mendeley customer_3d_phantom phantom_taobao
```

Prints a per-dataset table — never one merged number across annotation semantics — both
**before** and **after** post-processing, so the post-processing cannot hide the model's
behaviour. Sample overlays go to `results/unet/predictions/<run>_<dataset>/`.

| Flag | Default | What it does |
|---|---|---|
| `--ckpt` | required | Checkpoint path |
| `--dataset` | — | One or more datasets. Join with `+` to pool into one metric, e.g. `phantom_taobao+customer_3d_phantom` |
| `--min-area` | `4` | Joint class repair, radius-2 closing and hole filling; at most one region per class. Use 1 for very small targets; no opening. |
| `--max-dist` | `40` | Fragments further than this from the anchor (the largest component) are deleted as false positives; closer ones are kept as part of the same vessel. Only takes effect with `--allow-fragments` |
| `--allow-fragments` | off | Experimental distance gating for nearby fragments. Default: at most one region per class, possibly none. |
| `--vein-probe` | off | On `artery`-mode datasets (mendeley), measure how much vein the model predicts. The vein is really there but unannotated, and training never touches it, so this is a pure generalisation test |
| `--samples` | 10 | How many overlay images to save |
| `--no-save` | off | Skip writing overlays |
| `--no-fps` | off | Skip the throughput measurement |
| `--batch` | — | Batch size |

---

## infer.py — deployment inference

Self-contained: it imports only torch, numpy, cv2 and `model.py` / `postprocess.py`, so it can be copied
into the robot project as-is. No training-side imports, no `.env`, no dataset code.

```python
from models.unet.infer import UNetInferencer

inf = UNetInferencer("models/unet/checkpoints/unet_v10.pth")

# mask comes back on the CROP-FRAME grid — the same size you passed in
mask = inf.predict(cropped_gray_frame)          # uint8: 0=bg, 1=vein, 2=artery

# geometry in mm, using the calibration for the current depth setting
g = inf.measure(mask, mm_per_px=0.0832)         # 4 cm depth; 0.104 for 5 cm
print(g["artery"]["lateral_mm"], g["artery"]["depth_mm"], g["artery"]["radius_mm"])
```

| Method | Signature | Returns |
|---|---|---|
| `UNetInferencer(...)` | `(ckpt_path, device=None, fit_mode="croppad")` | Loads the checkpoint and reads `num_classes`, `base_ch`, `bilinear` from the config stored inside it |
| `.predict(...)` | `(crop_frame, threshold=0.5, return_prob=False, postprocess=True, min_area=4, closing_radius=2, fill_holes=True)` | Class-id mask on the crop-frame grid |
| `.predict_png(...)` | `(path, crop=None, **kw)` | Convenience wrapper that reads a PNG first |
| `.measure(...)` | `(mask_crop, mm_per_px, axis_x=None, skin_row=0, min_area=1)` | `{"vein": {...}, "artery": {...}}` with `lateral_mm`, `depth_mm`, `radius_mm` |
| `.clean(...)` | `(binary, min_area=1)` | Keep largest 8-connected region; preserve empty input; no morphology or hole filling. |
| `.mm_per_px_from_depth(...)` | `(depth_cm, crop_h)` — static | Calibration constant from the scanner's depth setting |

### The geometry contract — why the mask comes back on the crop frame

```
crop frame  --fit()-->  576x544  --model-->  mask  --unfit()-->  crop frame
                                                                 ^^^^^^^^^^
                                    geometry is measured HERE, where mm_per_px
                                    is defined. The model never sees a millimetre.
```

`fit()` only ever pads, centre-crops, or scales **both axes by one factor** — the aspect
ratio is never touched — and `unfit()` inverts it exactly. In the default `croppad` mode
there is no scaling at all, so one network pixel *is* one crop-frame pixel and the
inverse is a pure offset with zero resampling loss.

This matters for two reasons, both of which an earlier version got wrong:

1. **A non-uniform resize turns a round vessel into an ellipse.** Artery-versus-vein is a
   shape judgement — an artery is round and non-collapsible, a vein is flattened and
   compressible — so scaling the axes differently corrupts the one cue the 3-class model
   relies on.
2. **Geometry must be measured where `mm_per_px` is defined.** Measuring in 576×544
   network space and converting with a calibration defined for the crop frame cannot
   work if the two axes were scaled differently — no single scalar converts both.
   Measured on a real frame, the old version was **+27.6% off on lateral offset and
   +17.3% on radius**.

`fit` / `unfit_mask` are duplicated here rather than imported from
`data/pipeline/dataPrepare.py`, on purpose: this file must have no training-side
dependencies. They are required to stay bit-identical to the pipeline's versions.

---

## Related

- [`../../results/unet/README.md`](../../results/unet/README.md) — results, experiment log, paper material
- [`../../analysis/unet_analysis/`](../../analysis/unet_analysis/) — FPS benchmark, geometry, cross-domain probe, config summary
- [`../../data/README.md`](../../data/README.md) — datasets and the preprocessing pipeline

### Absent vessels, thin veins and display outputs

`predict()` now keeps at most one region per class; use `postprocess=False` for raw predictions. Probability output bypasses cleanup. Area thresholds use the input crop grid in deployment and the evaluation grid in test.py.

`predict_regions(frame)` returns `vein`, `artery`, their `vessel` union, `overlap`, `display_rgb` (blue vein / red artery / green overlap), and `vessel_rgb` (all union pixels green). Current softmax argmax classes are exclusive. `postprocess.av_outputs()` also accepts independent masks. Display id 3 is never a fourth model class or a training target.

Empty and artery-only full3 annotations remain valid under the existing partial-label loss. Cleanup cannot reject a lone false positive. test.py reports absent-class false-positive frame rates and pixel counts, and empty-frame false-positive rates for full3 only (lower is better), in addition to raw/cleaned Dice. These diagnostics do not alter training or checkpoint selection. Samples include a green `_vessel.png` union view.

Historical opening and 150/300 pixel cutoffs are removed; current joint repair uses local closing and hole filling. Earlier results require their original policy; see the validation/test diagnostic below for the latest behavior.

## Paired checkpoint evaluation

`compare.py --checkpoints <old.pth> <new.pth> --output <new-directory>` evaluates the same five test caches, reports raw and fixed-cleanup scores plus missing-class false positives, and produces JSON, CSV and an HTML image gallery. Run from the repository root. Optional flags: `--datasets`, `--split`, `--batch`. Existing report directories are never overwritten.

The completed [v10/v11 comparison](../../results/unet/BASELINE_COMPARISON_CN.md) uses best epochs 19/32. Final weights are stored together as `checkpoints/unet_v10.pth` and `checkpoints/unet_v11.pth`; original checkpoint configs remain intact for provenance.

## Joint postprocessing

The shared default repairs minority class pixels only in a vessel component with a single class anchor and ≥0.6 ownership; then radius-2 closing and hole filling complete each region. Components below 4 pixels are removed, at most one per class survives, and empty masks stay empty. Retained opposite-class pixels are protected. No opening is used. In inference, set `min_area=1` for tiny targets, or `closing_radius=0, fill_holes=False` to disable completion. Pixel parameters depend on the image grid.

`compare.py --postprocess joint` (default) reports raw / largest (legacy) / cleaned (new). Use `--postprocess largest` to reproduce the old report. Static `clean(binary)` remains deletion-only for geometry. See the [validation/test diagnostic](../../results/unet/BASELINE_COMPARISON_CN.md); this heuristic cannot guarantee correct class assignment.
