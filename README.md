# Flood Segmentation & Multimodal Representation Learning

## Overview

This repository contains a research-oriented pipeline for flood segmentation using SAR imagery and a multimodal self-supervised pretraining framework (DeCUR) using paired SAR and optical data.

The project is structured around two core components:

1. Supervised SAR Baseline
   - U-Net-based binary flood segmentation
   - Trained on Sen1Floods11-derived SAR data
   - Evaluated using strong hand-labeled masks

2. Multimodal Pretraining (DeCUR)
   - Paired SAR + optical representation learning using SEN12MS
   - Learns aligned latent spaces across modalities
   - Produces candidate encoder initializations (no downstream segmentation evaluation yet)

IMPORTANT:
Final experiments and results were produced on a remote workstation.
Local checkouts may not include all datasets, checkpoints, or logs.

---

## Repository Structure

    .
    ├── README.md
    ├── scripts/
    │   ├── build_sen12ms_combined_manifest.py
    │   ├── combine_filtered_south_asia.py
    │   ├── visualize_sen12ms_pair.py
    │   └── (legacy / experimental scripts)
    │
    ├── src/
    │   ├── Single_mode_baseline_src/
    │   │   ├── train.py
    │   │   ├── test.py
    │   │   ├── eval.py
    │   │   └── binary_mode.py
    │   │
    │   ├── Multi_modal_src/
    │   │   ├── train.py
    │   │   ├── eval.py
    │   │   └── DeCURLoss.py
    │   │
    │   ├── data_loader/
    │   │   ├── sar_dataset.py
    │   │   ├── optical_dataset.py
    │   │   ├── patch_dataset.py
    │   │   ├── fused_dataset.py
    │   │   ├── combined_manifest.py
    │   │   ├── collate.py
    │   │   └── docs/
    │   │
    │   └── models/
    │       ├── unet/
    │       │   ├── UNetModel.py
    │       │   └── UNetParts.py
    │       │
    │       └── decur/
    │           └── model.py
    │
    ├── datasets/
    │   ├── FilteredSouthAsia/
    │   │   ├── HandLabeled/
    │   │   │   ├── S1Hand/
    │   │   │   └── LabelHand/
    │   │   └── Combined/
    │   │
    │   └── SEN12MS/
    │       └── Combined/
    │
    ├── experiments/
    ├── runs/
    ├── checkpoints/
    ├── tests/
    └── docs/

---

## Core Workflows

### SAR Strong-Only Baseline (Final Supervised Model)

    python3 -m Single_mode_baseline_src.train \
      --img-dir "../datasets/FilteredSouthAsia/HandLabeled/S1Hand" \
      --mask-dir "../datasets/FilteredSouthAsia/HandLabeled/LabelHand" \
      --epochs 200 \
      --batch-size 16 \
      --learning-rate 7e-5 \
      --validation 10 \
      --classes 1 \
      --output-dir "experiments/hand_only_final"

- Binary flood segmentation
- Evaluated using strong hand labels
- Final supervised reference baseline

---

### Multimodal Pretraining (DeCUR)

    python src/Multi_modal_src/train.py \
      --combined-root datasets/SEN12MS/Combined \
      --output-dir runs/decur_smoke3 \
      --epochs 100 \
      --batch-size 64 \
      --num-workers 4 \
      --validation-split 0.1

- Paired SAR + optical training
- Learns aligned representations
- Outputs pretrained encoders (not segmentation models)

---

## Key Design Principles

### Supervision Strategy
- Strong labels are the final evaluation standard
- Weak labels are auxiliary only
- Weak-label validation can be circular and misleading

### Dataset Roles
- Sen1Floods11 (FilteredSouthAsia) → supervised segmentation
- SEN12MS → self-supervised multimodal pretraining

### Multimodal Alignment
- Based on manifest pairing and shape validation
- Does not guarantee pixel-perfect alignment

### DeCUR Interpretation
- Measures representation alignment
- Does not measure flood segmentation performance
- Downstream gains are future work

---

## Outputs

SAR baseline:
- Checkpoints: experiments/.../checkpoints/
- Logs: TensorBoard + CSVs
- Metrics: IoU, Precision, Recall, F1

DeCUR:
- Checkpoints: runs/.../checkpoints/
- Logs: alignment metrics (cosine similarity, consistency)

---

## Status

- SAR baseline (conceptually): complete
- SAR baseline (local runnable): may require path fixes
- DeCUR pretraining: implemented
- Downstream multimodal evaluation: not implemented
- Final results: stored in remote environment

---

## Notes for Handoff

- Final paper results were generated on a remote compute environment
- Local repository may:
  - lack datasets (SEN12MS)
  - lack final checkpoints
  - contain historical artifacts not tied to final results
- Use the wiki and exported results for authoritative evaluation

---

