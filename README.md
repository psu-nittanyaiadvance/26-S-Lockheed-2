# SAR Flood Segmentation (Baseline)

This repo is a minimal SAR-only baseline for flood segmentation using the
Sen1Floods11 subset filtered to the Indian subcontinent. All masks are
ground-truth (strong or weak labels only). Training uses 256x256 patches with
20% overlap via `PatchDataset`.

**Repo Structure**

```
.
├── datasets/
│   └── FilteredSouthAsia/
│       ├── WeaklyLabeled/
│       │   ├── S1Weak/
│       │   └── S1OtsuLabelWeak/
│       └── HandLabeled/
│           ├── S1Hand/
│           └── LabelHand/
├── scripts/
│   ├── export_patches.py
│   └── forward_pass_augmented.py
└── src/
    ├── data_loader/
    │   ├── sar_dataset.py
    │   ├── patch_dataset.py
    │   └── README.md
    └── train.py
```

**Quickstart (PowerShell)**

1. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

3. Train (patching is automatic in `src/train.py`):

```powershell
python src/train.py --img-dir "datasets/FilteredSouthAsia/HandLabeled/S1Hand" `
  --mask-dir "datasets/FilteredSouthAsia/HandLabeled/LabelHand" `
  --epochs 20 --batch-size 8 --learning-rate 1e-4 --validation 10 --classes 1
```

**Data Loader**

See `src/data_loader/README.md` for dataset usage and patching details.
