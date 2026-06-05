# SENTRY

This repository contains the reference implementation of SENTRY, a defense framework for trajectory-map multimodal recognition under adversarial perturbations. The model uses both raw trajectory sequences and rasterized distribution-map representations, and applies spatial-index-guided alignment to improve robustness when one or both modalities are perturbed.

## Overview

SENTRY contains three main components:

- A sequence encoder for trajectory inputs.
- A CNN encoder for rasterized map inputs.
- A spatial-index alignment module that selects informative trajectory keypoints and retains their physically co-located map pixels.

## Repository Structure

```text
.
├── main.py
├── README.md
└── models
    ├── Models_Fusion.py
    ├── Models_Image.py
    ├── Models_Sequence.py
    ├── SupCon_models.py
    └── utils.py
```

## Requirements

Install the required packages:

pip install torch numpy tqdm torchmetrics

## Data Preparation

The data loader expects preprocessed pickle files for trajectory samples, map images, auxiliary map channels, and trajectory-to-map pixel indices.

In `models/utils.py`, replace the placeholder paths:

traj_init_filename = "xxx.pickle"
map_filename = "xxx.pickle"
map_channel6_filename = "xxx.pickle"
traj_init_filename_geo = "xxx.pickle"

## Training

Run SENTRY with:

python ./main.py --Normalize_latlon --seq-attack-time 5 --map_attack_std 0.3 --backbone Estimator

## Notes

This anonymized release is intended for review and reproducibility. Dataset paths are intentionally replaced by placeholders and should be configured locally before running.
