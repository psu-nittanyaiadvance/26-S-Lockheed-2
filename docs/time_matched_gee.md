# Time-Matched GEE Downloads

## Overview
This pipeline builds a time manifest from the filtered Sen1Floods11 subset and downloads
Sentinel-1 (VV/VH) + Sentinel-2 (optical) stacks for the same region/time regime.

## 1) Build the time manifest
The manifest is derived from the filtered dataset folder and optional Sen1Floods11 metadata.
By default it scans `datasets/FilteredSouthAsia` for S1 image chips and computes WGS84
bounding boxes from the GeoTIFFs.

```powershell
python src/sen1floods11/metadata.py
```

Optional metadata path (CSV/JSON):

```powershell
python src/sen1floods11/metadata.py --metadata-path path/to/Sen1Floods11_metadata.csv
```

Output:
`data/derived/sen1floods11_time_manifest.csv`

## 2) Authenticate Earth Engine (one-time)

```powershell
earthengine authenticate
```

## 3) Run GEE downloads

```powershell
python scripts/gee_download_time_matched.py --config configs/gee_time_match.yaml
```

Outputs:
- GeoTIFFs: `data/derived/gee_time_matched/{split}/{sample_id}.tif`
- Manifest: `data/derived/gee_time_matched_manifest.csv`

## 4) Enable in the dataloader

```python
from src.data_loader.sar_dataset import SARDataset

ds = SARDataset(
    img_root="datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak",
    mask_root="datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak",
    ids_or_paths=[...],
    mode="weak",
    use_time_matched=True,
    time_matched_root="data/derived/gee_time_matched",
    time_matched_manifest="data/derived/gee_time_matched_manifest.csv",
    time_matched_missing_policy="zeros",
)
```

## Leakage avoidance option
If `exclude_sen1floods11_exact_tiles: true`, the download uses an expanded bbox
(`expand_bbox_factor`, default 2.0). This reduces exact-chip leakage by providing
more context around the chip. It does not mask the original chip footprint.
