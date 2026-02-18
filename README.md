# SAR Flood Segmentation

This repo focuses on data loading, time-matched SAR+Optical preparation, and
dataset guardrails for flood segmentation experiments. The U-Net and training
loop are owned by other teammates.

**Repo Structure**

```text
.
├── configs/
│   └── gee_time_match.yaml
├── data/
│   └── derived/
│       ├── sen1floods11_time_manifest.csv
│       ├── gee_time_matched/
│       │   └── <split>/<sample_id>.tif
│       └── gee_time_matched_manifest.csv
├── datasets/
│   └── FilteredSouthAsia/
│       ├── WeaklyLabeled/
│       │   ├── S1Weak/
│       │   └── S1OtsuLabelWeak/
│       └── HandLabeled/
│           ├── S1Hand/
│           └── LabelHand/
├── docs/
│   ├── ABLATIONS.md
│   ├── DATA.md
│   ├── DATALOADER.md
│   └── TIME_MATCHING.md
├── scripts/
│   ├── gee_download_time_matched.py
│   ├── smoke_test_time_matched.py
│   └── smoke_test_loader.py
└── src/
    └── data_loader/
        ├── sar_dataset.py
        ├── collate.py
        ├── splits.py
        ├── stats.py
        ├── discover_ids.py
        └── validate_dataset.py
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

3. (Optional) Download time-matched stacks from GEE:

```powershell
python scripts/gee_download_time_matched.py --config configs/gee_time_match.yaml
```

4. Smoke test the loader:

```powershell
python scripts/smoke_test_loader.py
```

5. Compute normalization stats (example):

```powershell
@'
from src.data_loader.sar_dataset import SARDataset
from src.data_loader.stats import compute_running_mean_std

ds = SARDataset(
    img_root="datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak",
    mask_root="datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak",
    ids_or_paths=["<sample_id_1>", "<sample_id_2>"],
    mode="weak",
    normalize_cfg="none",
    transforms=None,
)

mean, std = compute_running_mean_std(ds, max_samples=512)
print(mean)
print(std)
'@ | python -
```

**Docs**
- `docs/DATA.md`: dataset layout, manifests, ID conventions
- `docs/DATALOADER.md`: SARDataset, collate, splits, stats, validation
- `docs/TIME_MATCHING.md`: GEE downloader, band order, manifest checks
- `docs/ABLATIONS.md`: ablation grid and no-confound rules
