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

| File | Link |
|------|------|
| Sample dataset | [Google Drive](TODO) |
| Pretrained model (.pth) | [Google Drive](TODO) |

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
