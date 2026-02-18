# Data Loader Guide

This doc describes `SARDataset`, `default_collate`, split utilities, and normalization stats.

**SARDataset**

Location: `src/data_loader/sar_dataset.py`

Returns a tuple: `(image, mask, metadata)` where:
- `image`: `torch.float32` tensor with shape `[C, H, W]`
- `mask`: `torch.uint8` tensor with shape `[1, H, W]`, or `None` if `mode="none"`
- `metadata`: dict with at least `id`, `img_path`, `mask_path`
- When `use_time_matched=True`, metadata includes:
  `time_matched` (8-band tensor when available), `time_matched_status`, `time_matched_path`

Key parameters:
- `mode`: `weak`, `strong`, or `none`
- `ids_or_paths`: list of sample IDs or image paths
- `normalize_cfg`: `None`, `"none"`, or `{type: "zscore", mean: ..., std: ...}`
- `use_time_matched`: enable 8-band SAR+Optical stacks
- `time_matched_missing_policy`: `zeros`, `skip`, or `raise`

**default_collate**

Location: `src/data_loader/collate.py`

Stacks images and masks, keeps metadata as a list. It raises if a batch mixes labeled
and unlabeled samples.

**Splits**

Location: `src/data_loader/splits.py`

`make_split(ids, val_frac, seed)` sorts IDs, shuffles with a seed, then returns
train/val ID lists. Save these IDs to disk and reuse them across runs.

**Normalization Stats**

Location: `src/data_loader/stats.py`

`compute_running_mean_std(dataset)` computes per-band mean/std using Welford’s algorithm.
Use raw (unnormalized) inputs when computing stats.

Example:

```python
from src.data_loader.sar_dataset import SARDataset
from src.data_loader.stats import compute_running_mean_std

ds = SARDataset(
    img_root="datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak",
    mask_root="datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak",
    ids_or_paths=["tile_000123", "tile_000124"],
    mode="weak",
    normalize_cfg="none",
    transforms=None,
)

mean, std = compute_running_mean_std(ds, max_samples=512)
print(mean, std)
```

Command form:

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

**Validation Utilities**

Location: `src/data_loader/validate_dataset.py`

Run these for reliability checks:
- `validate_sample_shapes(ds, n=64)`
- `validate_time_matched(ds, n=64)`
- `validate_manifest_consistency("data/derived/gee_time_matched_manifest.csv")`
