# Vein Segmentation

Ultrasound vein segmentation system using UNet, built for the Clarius HD3 L7 ultrasound device. Covers the full pipeline: data collection, annotation, training, and real-time inference.

## Project Structure

```
vein_segmentation/
├── config.py                # Shared configuration (DataInfo: paths, dataset params)
├── handler_us.py            # Real-time ultrasound handler with ML inference
│
├── collection/              # Data acquisition & annotation
│   ├── collect_clarius.py   # Collect images via Clarius Cast API
│   ├── collect_a325.py      # Collect images via A325 video capture card
│   └── label.py             # Circular brush annotation tool (PySide6)
│
├── training/                # ML training pipeline
│   ├── prepare.py           # Data augmentation & train/val/test split
│   ├── train.py             # UNet training (BCE + Dice loss, AMP, W&B)
│   ├── test.py              # Model evaluation & inference
│   ├── predict.ipynb        # Prediction notebook
│   └── unet/                # UNet model architecture
│
├── gui/                     # Real-time segmentation GUI
│   ├── main.py              # Main GUI with live inference overlay
│   ├── us_thread.py         # Ultrasound streaming thread
│   ├── video_recorder.py    # Video capture with inference overlay
│   ├── pysidecaster.py      # PySide6 Clarius wrapper
│   ├── ml/                  # Inference engine (UNetInferencer)
│   └── assets/              # Icons & stylesheets
│
├── data/                    # Datasets
│   ├── phantom_taobao/      # Primary dataset (Clarius API collected)
│   └── customer_3d_phantom/ # Secondary dataset (A325 capture card)
│
├── checkpoints/             # Trained model weights (.pth)
├── lib/                     # Clarius Cast SDK (shared libraries & headers)
├── examples/                # Clarius API usage examples
└── ref/                     # Reference docs & images
```

## Pipeline

```
1. Collect    collect_clarius.py / collect_a325.py  ->  data/*/images/ + meta CSV
2. Annotate   label.py                              ->  data/*/masks/
3. Prepare    prepare.py                             ->  augmented NPZ files
4. Train      train.py                               ->  checkpoints/*.pth
5. Evaluate   test.py                                ->  metrics & visualizations
6. Deploy     gui/ or handler_us.py                  ->  real-time inference
```

## Installation

```bash
git clone --recursive https://github.com/alfredzhang98/vein_segmentation.git
cd vein_segmentation
pip install -r requirements.txt
```

### Download Data & Pretrained Weights

| Resource | Link |
|----------|------|
| Datasets (phantom + public) | [Google Drive](https://drive.google.com/drive/folders/1LPwezlAUmxWnUo791RM3OPAhGQHq5r0v?usp=sharing) |
| Pretrained models (.pth) | [Google Drive](https://drive.google.com/drive/folders/1YSGUPX_Ck_GIyM49vvtPI5_LDQHro7JR?usp=sharing) |

Place `.pth` files in `checkpoints/` (for training/test) or `gui/ml/checkpoints/` (for GUI inference).

## Quick Start

```bash
# Collect data (A325 capture card)
cd collection && python collect_a325.py

# Annotate masks
cd collection && python label.py customer_3d_phantom

# Prepare augmented dataset
cd training && python prepare.py

# Train
cd training && python train.py

# Test
cd training && python test.py
```

## Results

![segmentation_result](ref/mdimages/result_show.png)

```
Inference on Tesla V100-PCIE-32GB

[Inference] Total samples: 290
[Inference] Total time: 0.6602s
[Inference] Average per image: 0.0023s
[Inference] FPS: 439.25
[TEST] loss: 0.0155 | dice: 0.9866
```

## Real-time Demo

![real time demo](ref/mdimages/realtimedemo.gif)

## GUI Screenshots

<p align="center">
  <img src="ref/mdimages/gui1.png" alt="gui1" width="45%" />
  <img src="ref/mdimages/gui2.png" alt="gui2" width="45%" />
</p>

## Hardware

- **Ultrasound**: Clarius HD3 L7
- **Capture Card**: A325 (alternative to Clarius API)
- **Calibration**: 72.35 um/px (Clarius native) / 83.78 um/px (A325 cropped)

## Datasets

### Phantom Datasets (self-collected)

- **dataset1 (phantom_taobao)**: Commercial phantom from [Taobao](https://item.taobao.com/item.htm?id=762322402710)
- **dataset2 (customer_3d_phantom)**: Custom-made phantom (gelatin + agar + saline)

### Public Datasets

- **dataset3: Common Carotid Artery Ultrasound Images** ([Mendeley Data](https://data.mendeley.com/datasets/d4xt63mgjm/1))
  - 1100 ultrasound images + 1100 expert masks, resolution 709x749x3
  - Mindray UMT-500Plus with L13-3s linear probe, 11 subjects

- **dataset4: Carotid and Femoral Vessel Ultrasound Dataset** ([Kaggle](https://www.kaggle.com/datasets/fa8b3e1386722702d9c80a7d2d10d5d50eef20d14a604078b38d01c66fd9f356))
  - 2203 train + 911 validation images from 105 videos, 11 volunteers
  - Angell Pionner H20 Ultrasound Scanner

> **Note**: `pretrained_06042026.pth` was trained on dataset1 + dataset2 + dataset3 only. dataset4 was **not** used during training.

## License

This project's source code is released under the [MIT License](LICENSE).

**Third-party dataset licenses:**

| Dataset | License |
|---------|---------|
| dataset3 (Common Carotid Artery) | CC BY 4.0 |
| dataset4 (Carotid and Femoral Vessel) | CC BY-NC 4.0 (non-commercial use only) |

If you use dataset4 in your work, it is restricted to non-commercial purposes per its license.

## Citations

If you use the public datasets, please cite:

```bibtex
@misc{momot2022carotid,
  author = {Momot, Agata},
  title  = {Common Carotid Artery Ultrasound Images},
  year   = {2022},
  publisher = {Mendeley Data},
  doi    = {10.17632/d4xt63mgjm.1}
}
```

For dataset4, see: https://link.springer.com/chapter/10.1007/978-3-031-72083-3_61
