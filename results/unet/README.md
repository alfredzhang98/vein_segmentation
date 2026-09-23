# U-Net — results, experiment log and paper material

> Current results: [baseline comparison summary](BASELINE_COMPARISON_CN.md), including final metrics, postprocessing, limitations and reproduction commands. Git retains the summary and generation code; images, HTML and detailed CSV/JSON go under ignored `generated/`. Pre-existing results below remain historical v10 results.

> 中文版：[README_CN.md](README_CN.md)

This file merges what used to be `EXPERIMENTS.md` (experiment design and log) and
`PAPER_SEGMENTATION.md` (paper data points, failure analysis and manuscript drafts).
Everything here is specific to [`models/unet/`](../../models/unet/).

**Reference checkpoint**: [`models/unet/checkpoints/unet_v10.pth`](../../models/unet/checkpoints/)
— U-Net, 3 output channels (background / vein / artery), `base_ch=32`, **7.76 M
parameters**, dropout2d 0.3, epoch 19, best PRIMARY = 0.8622. Every number below comes
from this one checkpoint.

**Contents**

| Part | What it covers |
|---|---|
| [1. Results](#1-results) | Dataset splits, final test scores, the post-processing choice |
| [2. Failure analysis](#2-failure-analysis) | Where the vein score actually goes, and what that implies for future work |
| [3. Experiment log](#3-experiment-log) | Stage 1 binary domain transfer (Exp-A..E), stage 2 three-class joint training |
| [4. Paper material](#4-paper-material) | METHOD / RESULTS / DISCUSSION, original and revised |
| [5. Figures](#5-figures) | The plotting scripts in this directory |

---

# 1. Results

## 1.1 Where the numbers come from

**Mus-V (site / original paper)** — 11 healthy volunteers, Angell Pioneer H20 scanner,
arm and neck (carotid, femoral), **arteries and veins annotated separately**. 105 videos,
5–160 frames each. Official split: train 2203 / valid 911.

Consistency check against our code:

| Official | Ours | Match? |
|---|---|---|
| train 2203 | train **2203** | ✓ |
| valid 911 | val **441** + test **470** = **911** | ✓ official valid halved **by sequence** |
| 105 videos | 105 sequences | ✓ |

> We cannot split by frame: consecutive frames of one sweep are near duplicates, and a
> frame-level split would put the same vessel in both train and test. So we keep the
> official train/valid and halve valid **by sequence**, not by frame, into val/test.

**CCA (Mendeley d4xt63mgjm, Momot et al., 2022)** — **11 subjects**, Mindray UMT-500Plus
with an L13-3s linear probe, at least one examination per side per subject (2 in vascular
mode, 8 in carotid mode). **100 images each, 1100 total**, with technician-drawn,
expert-reviewed masks. Native resolution 709×749×3.

> We measure 130 distinct acquisition prefixes across 1100 images ✓, consistent with
> "11 subjects × several examinations each".

**Phantom** — two self-built phantoms with vessels at **different embedding depths**
(measured centroid depth: custom 206–225 px ≈ 17–19 mm; taobao 246–447 px), equivalent
radius 66.6–76.2 px ≈ 5.6–6.4 mm for custom and 29.6–115.8 px for taobao.

## 1.2 Dataset splits

**Splitting principle**: by **sequence / subject**, never by frame.

| Dataset | Subjects | Acquisition unit | Annotation semantics | Train | Val | Test | Total |
|---|---|---|---|---|---|---|---|
| **Mus-V** | **11** | 105 videos | background/vein/artery (complete) | **2203** | 441 | **470** | 3114 |
| **CCA (Mendeley)** | **11** | 11 subjects × 2 sides | common carotid only (jugular unlabelled) | **770** | 165 | **165** | 1100 |
| **Phantom (custom 3D)** | — | shallow vessel | vessel (type undetermined) | **21** | 4 | **5** | 30 |
| **Phantom (taobao)** | — | deep vessel | vessel (type undetermined) | **64** | 13 | 15 | 92 |
| **Total** | | | | **3058** | **623** | **655** | **4336** |

**Training samples per epoch** (each original is drawn repeatedly, freshly augmented each
time — see [`data/README.md`](../../data/README.md) §2.2):

| Dataset | Originals | ×factor | Samples/epoch | Share |
|---|---|---|---|---|
| Mus-V | 2203 | ×8 | 17624 | 70.9% |
| CCA | 770 | ×5 | 3850 | 15.5% |
| Phantom (taobao) | 64 | ×40 | 2560 | 10.3% |
| Phantom (custom) | 21 | ×40 | 840 | 3.4% |
| **Total** | | | **24874** | 100% |

> The phantom factor is high (×40) to counterweight its tiny share of the joint training
> set; otherwise the domain the robot actually deploys in gets diluted to under 1%.

The two phantom validation splits are **pooled** into one 17-image metric — scored apart,
a 4-image split is noise.

## 1.3 Final test results

**Configuration**: `min_area=150` + `keep_largest` (one closed region per class),
morphological open/close + hole filling. Identical to the deployed post-processing —
`clean()` in [`models/unet/infer.py`](../../models/unet/infer.py) and in
[`analysis/unet_analysis/predict_geometry.py`](../../analysis/unet_analysis/predict_geometry.py).

### The three results to report

| Domain | Metric | **Dice** |
|---|---|---|
| **Phantom (custom 3D)** | Vessel | **0.970** |
| **CCA (Mendeley)** | Artery | **0.956** |
| **Mus-V** | Artery | **0.882** |
| **Mus-V** | Vein | **0.617** |

### Full table (with the raw-argmax comparison)

| Dataset | | Raw argmax | After post-processing |
|---|---|---|---|
| Mus-V | Artery | 0.8752 | **0.8823** |
| Mus-V | Vein | 0.6095 | **0.6172** |
| Mus-V | Vessel (merged) | 0.8056 | **0.8047** |
| Phantom (custom) | Vessel | 0.9694 | **0.9696** |
| CCA | Artery | 0.9544 | **0.9555** |

Reproduce:

```bash
python models/unet/evaluate.py --ckpt models/unet/checkpoints/unet_v10.pth \
       --dataset musv mendeley customer_3d_phantom phantom_taobao --min-area 150
```

> `--min-area 150` is not optional: the flag **defaults to 0**, which disables
> post-processing. Without it the vein scores 0.6121 and the artery 0.8824 — the
> raw-argmax-with-keep-largest numbers, not the deployed ones.

### Why `keep_largest` (one closed region per class)

**Each class *is* one vessel** — this is not an imposed constraint, it is what the ground
truth says: the artery is a single connected component in **99.4%** of Mus-V frames, the
vein in 90–95%, the phantom in **100%**.

A distance gate (`max_dist=40`) by design keeps fragments near the main body, and in
practice leaves **3.6% of vein frames and 1.5% of artery frames** split into pieces.
**The cost is not merely cosmetic**: `measure()` computes a centroid, and **the centroid
of two separated pieces falls in the gap between them — a location where there is no
vessel at all, and that location is the robot's insertion target.**

**The Dice cost is zero**: vein 0.6187 → 0.6172 (−0.002), artery 0.8812 → 0.8823
(**+0.001**), both within noise.

> Note: `infer.clean()` and `predict_geometry.clean()` **were already keep-largest**;
> only `test.py` and the figures were not — meaning the evaluation and the figures were
> showing behaviour the deployed system never exhibits. They are now in step.

> `phantom_taobao` is excluded from the headline results (that set contains long-axis
> views, which are outside this task's usage scenario).

---

# 2. Failure analysis

## 2.1 What the vein's 0.617 actually is

**Two conventions** (all numbers from the same checkpoint as the results table):

| Convention | Vein | Artery |
|---|---|---|
| **All frames (the paper-reported figure)** | **0.617** (470 frames) | **0.882** (470 frames) |
| Excluding Dice = 0 frames | 0.753 (385 frames) | 0.906 (458 frames) |
| **Detected frames only** (ground truth non-empty **and** overlap non-zero) | **0.751** (381 frames = **81.1%**) | **0.905** (455 frames = 96.8%) |

> Use the **last row** in the paper body — its definition is the cleanest and the hardest
> to attack. The "excluding Dice = 0" row mixes in 4 frames of correct rejection (ground
> truth empty and the model reported nothing, scored 1.0 by convention), which lifts the
> score slightly. The two differ by only 0.002, but the rigour of the definition is worth
> the effort.

> **→ Once the model detects the vein, Dice is 0.751 (on 81.1% of frames), not 0.617.**
> The vein's overall score is not dragged down by imprecise boundaries but by an
> **18.1% all-or-nothing failure rate**. For downstream guidance these mean entirely
> different things: **once detected, the vein's outline quality (0.751) is close to the
> artery's (0.905)** — the gap is in *finding* it, not in *drawing* it.

**Breakdown of the Dice = 0 frames (vein, 85/470 = 18.1%)**:

| Cause | Frames | Detail |
|---|---|---|
| A. GT empty + model false-positive | **14** | vessel compressed to zero area by the probe; the model still reports one (median false area 3091 px) |
| B. GT present + complete miss | **51** | the whole vessel is not seen (median missed area 4138 px) |
| C. GT present + wrong location | **20** | a prediction exists, but with zero overlap |

The artery has only **12/470 (2.6%)** zero-Dice frames, and **no case A** — an artery
does not collapse, which is the physical root of the gap between the two classes.

## 2.2 Cause ①: the reference annotation itself is imprecise

> 🔒 **The annotation-defect statistics in this section are internal reference only —
> do not put them in the paper.** The paper states only the qualitative fact that the
> vein is harder to delineate and its reference annotation carries more label noise,
> **without the specific numbers**. Using numbers to accuse a public dataset of poor
> annotation quality shifts the reviewers' focus onto an argument with the dataset's
> authors, and our underlying evidence is visual inspection — not a fight worth picking.

Objectively measurable defects in the Mus-V ground truth:

| Defect | Vein | Artery | Ratio |
|---|---|---|---|
| Annotations with holes inside the lumen | **3.7%** (106/2842) | 0.1% (2/3102) | **37×** |
| Annotations containing <50 px specks | 0.4% | 0.03% | 13× |

> A lumen is anechoic, so there should be nothing inside it → **a hole is an unambiguous
> annotation defect**, not anatomy. The vein annotation carries this defect **37×** as
> often as the artery.

**More direct evidence: unlabelled empty-vein frames.** 18 test frames have zero vein
ground truth; looking at each one's neighbours:

| Sequence | Frame | This frame | Previous | Next | Verdict |
|---|---|---|---|---|---|
| 202304030805_33 | 48 | **0** | **5233** | **4993** | a 5000 px vessel cannot vanish and reappear in one frame → **missing label** |
| 202304030805_33 | 50 | **0** | 4993 | 5545 | **missing label** |
| 202304030805_33 | 33 | 0 | 1630 | 0 | probably a missing label |

**Of the 18 empty-vein frames, 6 (33%) look like missing labels and 12 (67%) are genuine
collapses.**

> **The scope of these numbers**: the above measures **machine-detectable defects**
> (lumen holes, specks, empty-frame contradictions). It does **not** measure boundary
> precision — and the "imprecise annotation" visible to the eye is mostly boundary. With
> no second set of annotations to compare against, boundary precision cannot be measured
> objectively, so **neither direction should be asserted**: we can say neither that
> annotation quality is the main cause nor that it is not. **State only what was
> measured, and let the numbers speak.**

## 2.3 Cause ②: a more compressed vein is recognised much less reliably

| Vein area (smaller = more compressed) | Mean recall |
|---|---|
| **Smallest 20%** (300–1512 px) | **0.456** |
| Middle | 0.830 |
| Largest 20% | 0.674 |

And **14 frames are compressed to zero area** (a mode the artery never exhibits).
**→ points directly at joint force + temporal modelling.**

## 2.4 How the zero-Dice frames are distributed

| | Vein |
|---|---|
| **Isolated dropouts** (neighbouring frames Dice > 0.5) | **4/84 = 5%** |
| **Failures in runs** (neighbouring frames also fail) | **80/84 = 95%** |

The sequence level says it more directly — failures are whole stretches, not scattered:

| Sequences | Mean Dice |
|---|---|
| Worst 5 | **0.104 / 0.284 / 0.324 / 0.480 / 0.502** |
| Best 5 | 0.762 / 0.794 / 0.795 / 0.811 / **0.845** |

→ **95% of zero-Dice frames have neighbours that also fail.** This is "some subjects fail
throughout", not "the occasional dropped frame". **It is the same conclusion as the
11-volunteer data ceiling, seen from the other side.**

## 2.5 Where future work should point

Both directions hold, and **both attack detection rather than delineation** — because the
post-detection 0.751 is already close to the artery's 0.905, so nearly all the remaining
headroom sits in that 18.1% of all-or-nothing failures.

1. **Force–image multimodal fusion.** From Mus-V's own description: the dataset "also
   records a **force sensor** mounted on the ultrasound probe… **to assist in identifying
   arteries and veins**". The **direct physical cause** of the vein disappearing is probe
   pressure flattening it (14 frames compressed to zero area; the artery never does
   this). Force directly encodes how hard the probe is pressing — an observable the
   dataset's authors provided expressly for this — and this work does not use it at all.
2. **Sequence-level temporal modelling.** Mus-V is continuous sweeps (105 videos, 5–160
   frames each), yet the model has no temporal context. Failures occur **in runs** (95%
   of zero-Dice frames have failing neighbours), but **even bad sequences contain visible
   frames** (the worst sequence still has ~11% of frames at Dice > 0.5), so a
   sequence-level tracker can propagate the vessel's location from those frames through
   the ones where it is momentarily compressed.
3. **More subjects** (11 → more) — the fundamental fix, and the most expensive.

> ⚠️ **Wording**: write "sequence-level model / tracking", **not "temporal smoothing" or
> "interpolating from neighbouring frames"**. 95% of zero-Dice frames have *immediate*
> neighbours that also fail, which falsifies naive neighbourhood smoothing; a tracker
> that sees the whole sweep is not subject to that limit (visible frames remain in the
> bad sequences to propagate from). A reviewer will spot the difference.

## 2.6 ⚠️ The vein probe on CCA — not usable as evidence in the paper

**What was measured**: CCA never supervises the vein (the loss depends on p₂ only, which
is numerically verified: holding p₂ fixed and pushing the vein's share from 0.498 to
1.000 leaves the loss bit-identical, Δ = 0). The model emits a >200 px region in the vein
channel on **104/165 = 63.0%** of CCA test frames, covering 14.1% of the predicted
foreground.

**This proves nothing.** CCA **has no vein annotation → no ground truth → no way to
verify those regions are the internal jugular vein**. They could be the jugular, or any
dark area, acoustic shadow, artefact, or a misclassified part of the artery. The probe
counted only "did the vein channel output anything", not "was the output correct".

> ❌ **Do not write**: "the model recovers the unannotated internal jugular vein in 63% of
> CCA test frames" — one reviewer asking "how do you know that is a vein?" ends it, and
> we have no answer.

**If you do want to use this point, three options** (by cost):

1. **Qualitative only, tied to the figure** (cheapest): give no percentage, just say
   "Fig. 6 (bottom row) shows the model additionally labelling an anechoic structure
   adjacent to the carotid as a vein, which CCA does not annotate." — a statement
   **readers can verify from the figure themselves**, requiring no ground truth.
2. **Manual sample verification** (recommended, affordable): take 20–30 random CCA frames
   and check by eye whether the vein-channel output lands on an anechoic lumen adjacent
   to the carotid; report "N/20 frames verified by inspection". That turns 63% into a
   defensible number.
3. **Drop the point** (safest): remove it from Results.

**The current Results draft follows option 1** (qualitative only, tied to Fig. 6).

---

# 3. Experiment log

## 3.1 Stage 1: binary domain transfer (Exp-A .. Exp-E)

**Goal**: find the best transfer strategy for using a public carotid dataset (Mendeley)
to improve vessel segmentation on the phantom. All experiments are evaluated on the
**phantom test set** (the target domain), so the comparison is fair.

> This stage's model is **binary** (vessel / background) and has been superseded by the
> three-class partial-label model below. It is kept because it is the direct evidence for
> *why* joint training was necessary, and because [`plot_results.py`](plot_results.py)
> plots exactly these numbers.

| Experiment | Training data | Val data | Strategy | Purpose |
|---|---|---|---|---|
| **Exp-A** (baseline) | phantom only | phantom | from scratch | current baseline |
| **Exp-B** | Mendeley only | **mendeley** | from scratch | size the domain gap |
| **Exp-C** | Mendeley + phantom mixed | phantom | from scratch | effect of pooling |
| **Exp-D stage1** | Mendeley pretrain | **mendeley** | pretraining | learn generic ultrasound features |
| **Exp-D stage2** | phantom finetune (all params) | phantom | two-stage transfer | expected best |
| **Exp-E stage2** | phantom finetune (frozen encoder) | phantom | two-stage + freeze | frozen vs full finetune |

> **On the val data**: Exp-B and Exp-D stage1 use the mendeley val split (matching their
> training domain, for early stopping and checkpoint selection); the rest use phantom
> (the target domain). **The final test is on the phantom test set in every case.**

### Results

| Experiment | Strategy | val Dice | val set | test Dice (phantom) | test Dice (mendeley) | Note |
|---|---|---|---|---|---|---|
| Exp-A | phantom only | 0.9087 | phantom | 0.8967 | 0.3147 | baseline; poor out-of-domain, as expected |
| Exp-B | Mendeley only | 0.9564 | mendeley | 0.4735 | 0.9509 | large domain gap: strong on source, weak on target |
| Exp-C | Mixed | 0.9257 | phantom | 0.9226 | 0.9479 | balanced across both; phantom ↑, Mendeley slightly ↓ |
| Exp-D v1 | Mendeley pretrain + full finetune (LR=3e-5) | 0.9185 | phantom | 0.9112 | 0.2735 | catastrophic forgetting; Mendeley generalisation lost |
| Exp-D v2 | Mendeley pretrain + full finetune (LR=1e-5) | 0.9201 | phantom | 0.9117 | 0.2682 | lower LR does not help; forgetting persists |
| Exp-E | Mendeley pretrain + frozen-encoder finetune | 0.9154 | phantom | 0.9041 | 0.2205 | freezing the encoder still forgets, and phantom is below Exp-C |

**Conclusion**: mixed training (Exp-C) is the only strategy where neither domain
collapses. Two-stage transfer suffers catastrophic forgetting whether the encoder is
frozen or not. That is the direct argument for joint training.

### `.env` quick reference

| Parameter | Exp-A | Exp-B | Exp-C | Exp-D stage1 | Exp-D stage2 | Exp-E stage2 |
|---|---|---|---|---|---|---|
| DATASET | phantom_taobao | mendeley | mixed | mendeley | phantom_taobao | phantom_taobao |
| VAL_DATASET | phantom_taobao | **mendeley** | phantom_taobao | **mendeley** | phantom_taobao | phantom_taobao |
| FREEZE_ENCODER | false | false | false | false | false | **true** |
| LEARNING_RATE | 3e-4 | 3e-4 | 3e-4 | 3e-4 | **3e-5** | 1e-4 |
| RESUME_PATH | — | — | — | — | stage1.pth | stage1.pth |

To run (after editing [`.env`](../../.env)):

```bash
python models/unet/train.py
python models/unet/evaluate.py --ckpt models/unet/checkpoints/<best>.pth \
       --dataset phantom_taobao --no-save
```

**Notes**

1. **Change `RUN_NAME` in `.env` before every experiment**, or the checkpoint filenames
   collide.
2. For **Exp-D/E stage 2**, set `RESUME_PATH` to the full path of the stage-1 checkpoint.
3. **`VAL_DATASET` selects the validation set**; the final test always runs
   `--dataset phantom_taobao`.
4. **The Mendeley NPZ must be built first**, or Exp-B/C/D raise `FileNotFoundError`.
5. W&B logs every run; `RUN_NAME` appears in the run name, which is how you tell them
   apart.

## 3.2 Stage 2: three classes + partial-label joint training

### Why a new design was needed

Once Mus-V is introduced, the four datasets **disagree about what a mask means** — this
is the core problem:

| Dataset | Mask values | What it knows | **What it does not know** | `label_mode` |
|---|---|---|---|---|
| `musv` | `{0,1,2}` | background / vein / artery, all of it | — | `full3` |
| `phantom_taobao` | `{0,255}` | where the vessel is | whether it is an artery or a vein (a phantom has no such notion) | `vessel` |
| `customer_3d_phantom` | `{0,255}` | where the vessel is | same | `vessel` |
| `mendeley` | `{0,255}` | where the **artery** is (common carotid) | where the jugular is — **unlabelled, swept into background** | `artery` |

The model **always emits 3 channels** (0 = background, 1 = vein, 2 = artery), but the
loss supervises only what each dataset **actually knows**
(`partial_label_loss` in [`models/unet/train.py`](../../models/unet/train.py)):

- **`full3` (Mus-V)** — standard 3-way CE + multiclass Dice, full supervision.
- **`vessel` (phantoms)** — supervises the marginal `p1+p2` only. That loss **depends on
  the sum alone**, so it places no constraint whatsoever on "is this tube an artery or a
  vein" — the assignment is free, decided by the prior learned from Mus-V.
- **`artery` (mendeley)** — supervises `p2` only. That loss **depends on p2 alone**, so it
  is invariant to how the remaining probability divides between background and vein —
  **an unlabelled jugular vein can be predicted as a vein at zero penalty.**

  > This is the crucial one. Taking the lazy route and supervising Mendeley's background
  > as `bg` under a 3-way CE would teach the model that veins look like background,
  > destroying the vein class learned from Mus-V. **It must be avoided.**

Everything is computed with `log_softmax` + `logsumexp`, making `log(p1+p2)` exact and
numerically stable. All three invariances are verified numerically (deviation ~1e-7).

**Deployment**: phantom / binary → `(p1+p2) > threshold`, the binary pipeline unchanged;
human anatomy / needs the distinction → `argmax(p0,p1,p2)`. **One model, two uses.**

### Data preparation

```bash
python data/pipeline/dataPrepare.py --dataset musv --no-png
# 105 sweeps / 3114 frames → train 80 seq (2203 frames) / val 12 seq (441) / test 13 seq (470)
```

> **Split by sequence, never by frame.** Consecutive frames of one sweep are near
> duplicates; a random frame-level split would put the same vessel in both train and
> test, inflating Dice. Mus-V ships an official train/valid split — keep it, then halve
> the official valid's 25 sequences **by sequence** into val / test.

### Running

Single GPU (an H200 takes one card at a time, `MAX_GPUS` defaults to 1; multi-GPU
DataParallel is the only configuration ever observed to turn val loss into NaN):

```bash
# Exp-G — Mus-V only, 3 classes. Serves as the generalisation-probe model
EPOCHS=200 EARLY_STOP_PATIENCE=20 NUM_CLASSES=3 \
DATASET=musv VAL_DATASET=musv,phantom_taobao,mendeley \
BATCH_SIZE_PER_GPU=16 LEARNING_RATE=0.0003 RESUME_PATH= FREEZE_ENCODER=false \
RUN_NAME=expG_musv_3class python models/unet/train.py

# Exp-H — four-dataset partial-label joint training (the headline result)
EPOCHS=200 EARLY_STOP_PATIENCE=20 NUM_CLASSES=3 \
DATASET=musv+phantom_taobao+mendeley+customer_3d_phantom \
VAL_DATASET=musv,phantom_taobao,mendeley \
BATCH_SIZE_PER_GPU=16 LEARNING_RATE=0.0003 RESUME_PATH= FREEZE_ENCODER=false \
RUN_NAME=expH_joint_partial_label python models/unet/train.py
```

Join datasets with `+` to train on the union; each sub-dataset carries its own
`label_mode`, and the loss routes every sample to the right supervision mode
automatically.

**Checkpoint / early-stop metric** (`primary_score` in `train.py`): if the primary
validation set is fully annotated (Mus-V) → use `(dice_vein + dice_artery)/2`, because
separating the two *is* the task and a merged `dice_vessel` stays high even when the
model confuses them; otherwise fall back to `dice_vessel`.

## 3.3 Generalisation probe

```bash
python analysis/unet_analysis/probe_generalization.py \
       --ckpt models/unet/checkpoints/<x>.pth \
       --datasets musv mendeley phantom_taobao
```

**Mus-V is the yardstick** — every frame labels both artery and vein, so for any model
(even one that has never seen a vein label) it can quantify: *of the pixels that really
are vein, how many did the model call vessel?* That turns "let's eyeball it" into a
number. The script handles both the old binary checkpoints (`n_classes=1`) and the
3-class model.

### Result: Exp-B (mendeley only, has only ever seen arteries) → Mus-V test

| True class | Called background | Called vessel |
|---|---|---|
| Background | 99.94% | 0.06% |
| **Vein** | 97.12% | **2.88%** |
| **Artery** | 59.22% | **40.78%** |

**A model trained only on artery data does not generalise to veins**: its recall on
arteries is **14×** its recall on veins. What it learned is not a generic "anechoic
lumen" feature but something **artery-specific**. The overlays agree: the green
prediction lands almost exclusively on the blue artery and skips the red vein.

> This is the direct justification for Mus-V + partial-label joint training: you cannot
> *grow* a vein class out of Mendeley.

---

# 4. Paper material

Target journal: TMECH. Each section below gives the original and the revised text, with
the rationale for the change.

## 4.1 METHOD — original

> For ultrasound-based vessel localisation, a U-Net-based segmentation module was adapted for vessel-mask extraction and downstream geometric guidance [19]. The aim of this module is not to introduce a new segmentation architecture, but to provide a reliable vessel-region estimate from which the centroid, lateral offset, depth, and radius can be extracted. The model is trained using our custom vascular phantom dataset together with the public Mendeley carotid artery dataset (CCA) [20]. CCA is used as an auxiliary domain because the task focuses on vessel-region segmentation rather than artery–vein classification. In short-axis ultrasound, the internal jugular vein can appear approximately circular under Trendelenburg positioning, making carotid artery images useful for learning generic vessel-boundary features despite the anatomical difference. The network is trained with a compound BCE–Dice loss [21],
>
> L = L_BCE + (1 − (2 Σ pᵢgᵢ + ε) / (Σ pᵢ + Σ gᵢ + ε))   (1)
>
> where pᵢ is the predicted vessel probability, gᵢ is the binary ground-truth label, and ε is a small constant for numerical stability. As shown in Fig. 3(a), the probability map is dichotomised into a binary mask and refined by morphological cleaning. Image-moment analysis and the calibrated ultrasound pixel scale are then used to estimate the vessel centroid, lateral offset δ, depth d, and radius r for downstream alignment and insertion guidance.

## 4.2 METHOD — revised

> For ultrasound-based vessel localisation, a U-Net-based segmentation module was adapted for vessel-mask extraction and downstream geometric guidance [19]. The aim of this module is not to introduce a new segmentation architecture, but to provide a reliable vessel-region estimate from which the centroid, lateral offset, depth, and radius can be extracted.
>
> The module is trained jointly on two custom vascular phantoms with vessels at different embedding depths, the public Mus-V vascular dataset [X] (11 volunteers, arteries and veins annotated separately), and the public Mendeley common carotid artery dataset (CCA) [20] (11 subjects, carotid artery only). These sources disagree on what their masks mean: Mus-V annotates the vein and the artery separately; the phantoms annotate a tube whose type is undefined; CCA annotates only the carotid artery, while the internal jugular vein is frequently within the field of view but left unlabelled. Pooling them under a single objective would therefore supervise CCA's unlabelled jugular vein as background, penalising a correct vein prediction wherever that vessel appears.
>
> The network therefore emits three channels — background, vein and artery, with softmax probabilities p₀, p₁ and p₂. Each dataset's annotation resolves these three classes only up to a partition 𝒢 of the class set: Mus-V separates all three, so 𝒢 = {{bg}, {vein}, {artery}}; the phantoms mark a tube of undetermined type, so 𝒢 = {{bg}, {vein, artery}}; CCA marks the carotid artery while leaving the internal jugular vein unlabelled inside the background region, so 𝒢 = {{bg, vein}, {artery}}. Writing q_g = Σ_{c∈g} p_c for the marginal probability of a group g ∈ 𝒢 and y_g ∈ {0,1} for its annotation, the compound cross-entropy–Dice objective [21] is applied to these marginals:
>
> **L = − (1/|Ω|) Σᵢ∈Ω Σ_{g∈𝒢} y_{g,i} log q_{g,i}  +  (1/|𝒢⁺|) Σ_{g∈𝒢⁺} (1 − (2 Σᵢ q_{g,i} y_{g,i} + ε) / (Σᵢ q_{g,i} + Σᵢ y_{g,i} + ε))   (1)**
>
> where Ω denotes the pixels of an image, ε is a small constant for numerical stability, and 𝒢⁺ are the foreground groups — the group containing the background class is excluded from the Dice term, since background covers ~96% of pixels and is predicted near-perfectly, so retaining it would only dilute the gradient on the vessel groups that term exists to drive. A batch may mix the three annotation types; the corresponding losses are combined as a sample-count-weighted mean.
>
> For a two-group partition — the phantoms and CCA — the first term is exactly the binary cross-entropy L_BCE(q, y) and the second reduces to a single Dice term, so (1) recovers the conventional compound BCE–Dice loss evaluated on the supervised marginal. Its effect is to leave unsupervised distinctions free: for the phantoms the objective depends on p₁ + p₂ alone, so the division between vein and artery is unconstrained and is resolved by the prior learned from Mus-V; for CCA it depends on p₂ alone and is invariant to how the remaining probability divides between background and vein, so an unlabelled jugular vein may be predicted at no cost. For Mus-V all three groups are singletons and (1) becomes a three-way cross-entropy together with the Dice term evaluated on the vein and the artery channel and averaged over the two. All terms are evaluated in the log domain via log-softmax and log-sum-exp, making log(p₁ + p₂) exact and numerically stable.
>
> As shown in Fig. 3(a), the probability map is dichotomised into a binary mask and refined by morphological cleaning: opening and closing suppress speckle, interior holes are filled — a vessel lumen is anechoic, so an enclosed hole is a segmentation artefact and, left in place, biases the area-equivalent radius — and components lying far from the principal vessel are discarded as false positives. Image-moment analysis and the calibrated ultrasound pixel scale are then used to estimate the vessel centroid, lateral offset δ, depth d, and radius r for downstream alignment and insertion guidance. The network itself is purely a pixel-to-pixel map; the physical scale enters only after the predicted mask is returned to the native image grid, so a single calibration constant converts both axes exactly.

**What changed and why**

| Item | Original | Revised | Reason |
|---|---|---|---|
| Datasets | phantom + CCA | + **Mus-V** | the vein class can only be learned from Mus-V |
| Output | 1 channel (vessel) | **3 channels** (bg/vein/artery) | the task now distinguishes artery from vein |
| Role of CCA | "CCA is usable because we do not classify artery vs vein" | "CCA labels only the artery, so its unlabelled vein must cost nothing" | **the original logic no longer holds** — we now do classify them |
| Formula | BCE–Dice on the vessel probability p | **CE–Dice on the group marginals q_g of the partition 𝒢 the annotation resolves** | **keeps one formula**: for a two-group partition the first term degenerates exactly to L_BCE and the second to a single Dice, i.e. the original — a generalisation, not a replacement |
| Post-processing | "morphological cleaning" | explicit: open/close + **hole filling** + distant-island removal | hole filling changes the radius |
| — | — | new: the model carries no physical scale | answers reviewer concerns about calibration |

## 4.3 RESULTS — original

> To verify the ultrasound segmentation module used in the subsequent guidance experiments, the selected U-Net model was trained jointly on the custom vascular phantom dataset and the public CCA dataset. The model achieved Dice scores of 0.923 and 0.948 on the held-out phantom and CCA test sets, respectively. As shown in Fig. 6, the predicted vessel regions closely matched the manual annotations in both domains, despite the different ultrasound appearances of the phantom and anatomical images. The jointly trained model was therefore used to provide vessel masks for estimating the vessel centre and radius in the following insertion experiments.

## 4.4 RESULTS — revised

> To verify the ultrasound segmentation module used in the subsequent guidance experiments, the U-Net model was trained jointly on the vascular phantoms, Mus-V and CCA, and evaluated on held-out test sets held disjoint at the sequence and subject level.
>
> On the phantom — the domain the guidance experiments are performed in — the model segmented the vessel region with a Dice of **0.970**. On the two anatomical datasets it localised the artery with a Dice of **0.956** on CCA and **0.882** on Mus-V, confirming that a single set of weights transfers across ultrasound appearances as different as a gel phantom and a human neck. For these classes the predicted boundaries follow the manual annotations closely (Fig. 6). End to end — preprocessing, inference, morphological refinement and moment analysis — the module runs at **5.6 ms per frame** (5.7 ms at the 95th percentile) at batch size 1 on an NVIDIA H200 NVL, i.e. **179 fps**, well above the ultrasound frame rate, so segmentation does not bound the guidance loop.
>
> Mus-V additionally annotates the vein, on which the model reached a Dice of **0.617**. This figure is governed by detection rather than delineation: on the **81.1%** of frames where the vein is detected, the Dice is **0.751**, against **0.905** for the artery under the same criterion — once the vein is found, it is outlined nearly as accurately.
>
> Two factors account for the remaining gap. The first is that the vein is intrinsically harder to delineate: it is thin-walled and low-contrast against the surrounding tissue, so its boundary is less sharply defined than an artery's — for the annotator as much as for the network. The second is compression. A vein collapses under probe pressure, and recognition degrades as it flattens: recall falls to 0.46 on the most flattened quintile of vein cross-sections against 0.83 at mid-range, and in 14 test frames the vein is compressed to zero cross-section entirely — a mode the non-collapsible artery never exhibits. Mus-V's 105 videos originate from only 11 volunteers, which bounds how much of that deformation a single-frame model can learn; the train–validation Dice gap stands at 0.03 for the artery against 0.24 for the vein.
>
> A qualitative observation supports the partial-label formulation. CCA never supervises the vein — by construction, (1) is invariant to it there — yet in Fig. 6 (bottom row) the model additionally labels an anechoic structure adjacent to the carotid, which we confirmed by inspection to be the internal jugular vein, and which the CCA annotation does not cover. As CCA provides no vein ground truth, this is reported qualitatively rather than scored.
>
> Since the guidance task requires the vessel region and its geometry, and vessel-type discrimination is a secondary capability, the jointly trained model was used to provide vessel masks for estimating the vessel centre and radius in the following insertion experiments.

**What changed**

| Item | Original | Revised |
|---|---|---|
| Numbers | phantom 0.923 / CCA 0.948 | **phantom 0.970 / CCA 0.956 / Mus-V artery 0.882 / vein 0.617** + FPS |
| Structure | one flat paragraph | **strengths first** (phantom 0.970 + artery in both domains), vein in its own paragraph |
| Vein | — | **split into detection and delineation**: 0.751 once detected, close to the artery's 0.905 |
| Attribution | — | physical root cause (compressible, 14 frames at zero area) + data ceiling (11 volunteers) + gap 0.03 vs 0.24 |
| Real-time | — | **5.6 ms/frame, 179 fps (H200 NVL, batch = 1)** — TMECH is a mechatronics journal; this is mandatory |
| Generalisation | — | **qualitative** observation tied to Fig. 6 only; **no percentage** (CCA has no vein ground truth, so it cannot be quantified) |

**Writing technique**

1. **Strengths first, weakness second.** Establish phantom 0.970 and the artery in both
   domains (0.956 / 0.882), then the real-time figure. The reader's anchor becomes "this
   module is reliable and deployable", so the vein's 0.617 does not dominate the first
   impression.
2. **Decompose the 0.617.** "Dice 0.617" sounds like "the outline is bad"; in fact it is
   **0.751 on the 81.1% of frames where the vein is detected**, with the rest being
   all-or-nothing failures. **This is not spin — it is the more accurate description**,
   and 0.751 survives a reviewer recomputing it.
3. **Use "dominated by detection rather than delineation".** One phrase moves the reader
   from "the model draws badly" to "the model occasionally cannot find it", which means
   something entirely different downstream.
4. **Attribute to physics, not to a model defect.** "14 frames compressed to zero
   cross-section" is a **verifiable objective fact**, far stronger than "veins are
   harder".
5. **Close with a qualitative observation and say outright it is not quantified.**
   Declaring the absence of ground truth yourself reads as restraint, and the reader can
   verify the statement from the figure.

> **Do not ① **: do not report 0.751 without 0.617. 0.617 is the honest figure over all
> test frames and a reviewer will ask which convention you used. **Give both and state
> the difference** — far more credible than hiding one.
>
> **Do not ② **: do not write "recovers the internal jugular vein in 63% of CCA frames".
> The probe counted whether the vein channel output anything, **not whether the output
> was correct** — CCA has no vein ground truth, so the claim cannot be supported. To
> quantify it you need manual sampling (see §2.6, option 2).

## 4.5 DISCUSSION — new paragraphs (paste-ready)

> The residual error on the vein lies in detection rather than delineation. Once detected, the vein is delineated at a Dice of 0.751 against the artery's 0.905; what separates the two classes is that the vein is missed outright on 18.1% of frames. Those misses are not distributed uniformly — 95% fall in runs whose neighbouring frames fail as well, and five of the thirteen test sequences account for most of them — and they track vessel compression directly: recall drops to 0.46 on the most flattened quintile of veins. Two directions follow, both attacking detection, and neither exploited by the present single-frame model.
>
> First, the failures have a single physical driver. A vein is thin-walled and collapses under probe pressure, vanishing entirely in 14 of the test frames — a mode the non-collapsible artery never exhibits. Mus-V records synchronised probe-force measurements expressly to help distinguish arteries from veins, and force is precisely the quantity that governs the collapse. Conditioning the segmentation on force therefore addresses the failure at its source rather than at its symptom.
>
> Second, the data are acquired as continuous probe sweeps, yet each frame is currently segmented in isolation. Because the failures occur in runs, frame-to-frame smoothing would be of little help; a sequence-level model, however, can exploit the frames within a sweep in which the vein remains visible — present even in the weakest sequences — and propagate its location through the frames in which it is momentarily compressed. Jointly conditioning on force and sequence context targets the same underlying event from two directions: force says when the vein is being flattened, and the sweep says where it was before it flattened.
>
> A sequence-level formulation would additionally confer robustness to isolated labelling errors, to which a per-frame objective is fully exposed. Together these routes address the all-or-nothing failures that presently bound the vein score, without disturbing the delineation quality, which is already adequate.

**Why it is written this way**

| Device | Effect |
|---|---|
| Open with **0.751 vs 0.905** | frames the problem immediately: it is about *finding*, not *drawing* |
| State the 95% run-failure rate outright | **volunteers the weakness** before a reviewer finds it, and doubles as the argument for why naive smoothing is not the answer |
| "a mode the artery never exhibits" | pins the low vein score to physics via a **control fact**, not to a model defect |
| "addresses the failure at its source rather than at its symptom" | elevates the force modality from "another input" to a **causal** fix |
| "frame-to-frame smoothing would be of little help; a sequence-level model, however…" | **rejects the weak version before proposing the strong one** — shows we know where the boundary is, which raises credibility sharply |
| Close with "without disturbing the delineation quality, which is already adequate" | the headroom is bounded and identified; this is not "the model needs rebuilding" |

## 4.6 Loss / checkpoint logic, in one sentence each

**Loss**:
> Each dataset supervises only the marginal its annotation can actually determine — the
> phantoms constrain the vein+artery sum only (the split between them stays free), CCA
> constrains the artery channel only (an unlabelled jugular vein may be predicted at zero
> penalty), Mus-V supervises all three — which lets one model train jointly on four
> datasets whose annotation semantics contradict each other, without them destroying one
> another.

**Checkpoint**:
> Score each validation set by deployment priority (phantom 0.45 / Mus-V 0.45 / CCA 0.10)
> and weight them, using for each set only the metric its annotation can support (Mus-V:
> (vein+artery)/2; phantom: vessel Dice; CCA: artery Dice), so the selected checkpoint is
> not systematically misled by a dataset's missing annotation.

## 4.7 Model configuration table

| Parameter | Value |
|---|---|
| Architecture | U-Net (3-class, partial-label) |
| Base width | 32 |
| Parameters | 7.76 M |
| Input | 576 × 544, grayscale |
| Dropout2d | 0.3 |
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻³ |
| Batch size | 16 |
| Loss | Compound grouped-CE–Dice on the marginals the annotation resolves, Eq. (1) (= BCE–Dice for the two-group datasets) |

> Generate this table rather than typing it — every number is read from the checkpoint:
> ```bash
> python analysis/unet_analysis/summarize_model.py --ckpt <x>.pth --latex
> ```

---

# 5. Figures

The scripts in this directory. Run all of them from the repository root.

| Script | Output | Notes |
|---|---|---|
| [`plot_segmentation_samples.py`](plot_segmentation_samples.py) | `figures/segmentation_samples_<checkpoint>.png` | **Five-domain qualitative figure.** One row per dataset with final joint postprocessing. Checkpoint-named output; existing files are never overwritten. |
| [`plot_results.py`](plot_results.py) | `figures/exp_results.{pdf,svg,png}` | Grouped bar chart of Exp-A..E. **Note: this plots the stage-1 binary experiments of §3.1**, not the current 3-class model |
| [`figure_style.py`](figure_style.py) | — | Shared matplotlib styling; produces no figure of its own. Imported by both scripts above |

```bash
# The paper's qualitative figure
python results/unet/plot_segmentation_samples.py \
       --ckpt models/unet/checkpoints/unet_v11.pth \
       --n 2 --seed 42

# Default: PNG at 200 dpi. --preview: 150 dpi; --formats png pdf svg: optional export formats.
```

`FigureConfig` in `figure_style.py` has three presets: the default (large type, still
legible after IEEE two-column downscaling), `ieee_strict()` (8–10 pt, for already-final
layout) and `presentation()` (slides / posters). Every figure is authored at 12.0" width
and scaled to the paper column by the same factor, so the text in all of them prints at
the same physical size — do not change one figure's width on its own.

## Training monitoring

The standard-library-only `models/unet/monitor.py` polls a training PID, its GPU allocation and checkpoint updates every 60 seconds. It exits when the training process ends and never modifies or restarts training.

```bash
python -u models/unet/monitor.py --pid <training-PID> \
  --checkpoint-dir models/unet/checkpoints/<version> \
  --train-log results/unet/runs/<run-name>/train.log
```

Training and its background monitor have ended; historical records are in `results/unet/runs/unet_v11_balanced/monitor.log` with `tail -f`. This run's `tee` has no training log open, so batch progress, losses and the exit reason were available only in the original terminal; the user confirmed successful early stopping at epoch 62. A checkpoint's epoch is the best saved epoch, not necessarily the current epoch. For future runs, create the log directory first and verify that `tee` opens its output successfully.

Outputs default to `segmentation_samples_<checkpoint>.png` with a JSON manifest of cache indices and per-image scores. Use `--datasets`, `--n`, `--seed` or a new `--out` prefix. All five datasets are included by default, preferring labelled visible vessels without consulting predictions. Existing outputs are rejected. New checkpoint-named artifacts are Git-ignored except the retained final `segmentation_samples_unet_v11_showcase.png/.json`; historical `segmentation_samples.png/.svg` remain intact.

Use `--datasets musv mendeley pmc9883282 customer_3d_phantom --n 2 --showcase-datasets musv mendeley pmc9883282 --showcase-min-dice 0.90 --out results/unet/figures/segmentation_samples_unet_v11_showcase` for diverse success examples in those three domains, requiring each labelled class Dice ≥0.90. Select the strongest first example, then maximize distance based on 45% class-mask Dice distance, 45% relative class-area difference and 10% normalized 32×32 image RMSE. This selection is disclosed here and in the JSON manifest; it does not imply different subjects. The layout matches historical `segmentation_samples`, without an added title or footer; retain this selection disclosure when reusing the figure. Phantom rows retain fixed-seed selection. Original figures and full-test metrics remain unchanged.
