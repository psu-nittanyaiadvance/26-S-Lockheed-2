# Data Loader (src/data_loader)

This repo's core functionality lives in `src/data_loader`. The loader is built for
Sentinel-1 SAR tiles with optional masks and optional time-matched SAR+Optical
stacks. It is intentionally strict about shapes and dtypes so downstream models
see consistent tensors.

**Core Dataset: `SARDataset`**

`SARDataset` returns a 3-tuple: `(image, mask, metadata)`.
- `image`: `torch.float32` with shape `[C, H, W]` read via rasterio.
- `mask`: `torch.uint8` with shape `[1, H, W]`, or `None` when `mode="none"`.
- `metadata`: dict with `id`, `img_path`, `mask_path`, plus time-matched fields
  when enabled.

`mode` controls label behavior:
- `mode="weak"`: masks are read from `mask_root` (weak labels).
- `mode="strong"`: masks are read from `mask_root` (hand labels).
- `mode="none"`: no masks returned; `mask_root` is not required.

ID/path resolution:
- `ids_or_paths` can be IDs like `tile_000123` or explicit `.tif/.tiff` paths.
- `ids_are_paths` is inferred if any item is absolute, has a tif suffix, or
  contains a path separator.
- For ID inputs, images are resolved to `img_root/<id>.tif` (fallback to `.tiff`).
- For path inputs, images use the given path and masks use `mask_root/<stem>.tif`.
- The dataset does not map weak<->strong IDs; it only uses the provided list.

Image and mask loading:
- Images are read as `[C,H,W]`, cast to `float32`, and optionally log-transformed
  (`log_transform=True` uses `log1p(abs(x))`).
- Masks are read from band 1, validated for shape match, and binarized
  (`mask > 0`) into `[1,H,W]` with `uint8` dtype.

Normalization:
- `normalize_cfg=None` or `"none"`: no normalization.
- `normalize_cfg={"type": "zscore", "mean": ..., "std": ...}`: per-band z-score
  with strict checks for channel count and positive std values.
- For raw stats computation, create the dataset with `normalize_cfg="none"` and
  `transforms=None`.

Validation and filtering:
- `validate=True` validates each sample at init, dropping unreadable rasters or
  mismatched mask sizes with a warning. If all samples fail, it raises.
- Errors during `__getitem__` are explicit (e.g., missing masks in labeled mode).

Transforms:
- `transforms` is called as `transforms(image, mask, metadata)`.
- It must return `(image, mask, metadata)` or `(image, mask)`; other outputs
  raise a `ValueError`.

**Time-Matched SAR+Optical Stacks**

Enable time-matched stacks with `use_time_matched=True`. This augments
`metadata` with:
- `time_matched`: `torch.float32` stack of shape `[8,H,W]` (VV, VH + 6 S2 bands).
- `time_matched_path`: resolved path or `None`.
- `time_matched_status`: status string from the manifest or a fallback value.

Path resolution and status handling:
- The loader reads `time_matched_manifest` (CSV) when present and indexes by
  `sample_id`. If a row has `status="ok"` and a valid `path`, it uses that path.
- Otherwise it falls back to `time_matched_root/<split>/<id>.tif`, where `split`
  is inferred from the image path (`WeaklyLabeled`, `HandLabeled`, else `all`).
- Manifest statuses like `missing_s1`, `missing_s2`, `missing_both`, or `error`
  are treated as missing for loading.

Missing policy (`time_matched_missing_policy`):
- `zeros`: return an all-zero `[8,H,W]` stack (warning emitted once).
- `skip`: drop samples without a valid time-matched stack at init time.
- `raise`: raise at `__getitem__` when a stack is missing.

**Batching (`default_collate`)**

`default_collate` stacks images into `[B,C,H,W]` and masks into `[B,1,H,W]`.
It preserves metadata as a list and raises if a batch mixes labeled and
unlabeled samples.

**Dataset Utilities**

ID discovery and pairing (`discover_ids.py`):
- `list_ids_from_dir(img_root)`: sorted stems for `.tif/.tiff` files.
- `paired_ids(img_root, mask_root)`: intersection of image/mask IDs plus a
  missing-count report.

Deterministic splits (`splits.py`):
- `make_split(ids, val_frac, seed)` sorts IDs, applies a seeded shuffle, and
  ensures at least one validation sample when `val_frac > 0`.

Normalization stats (`stats.py`):
- `compute_running_mean_std(dataset, max_samples=None)` uses Welford's
  algorithm for streaming per-band mean/std on `[C,H,W]` images.

Validation helpers (`validate_dataset.py`):
- `validate_sample_shapes(ds, n=64)` checks image/mask shapes, dtypes, and
  finite values.
- `validate_time_matched(ds, n=64)` checks time-matched stacks and missing-policy
  behavior.
- `validate_manifest_consistency(path)` validates required manifest columns and
  allowed status values.

**Usage Example**

```python
from pathlib import Path
from torch.utils.data import DataLoader

from src.data_loader import (
    SARDataset,
    compute_running_mean_std,
    default_collate,
    make_split,
)

img_root = Path("datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak")
weak_mask_root = Path("datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak")

ids = ["tile_000123", "tile_000124", "tile_000125", "tile_000126"]
train_ids, val_ids = make_split(ids, val_frac=0.2, seed=1337)

train_raw = SARDataset(
    img_root=img_root,
    mask_root=weak_mask_root,
    ids_or_paths=train_ids,
    mode="weak",
    normalize_cfg="none",
    log_transform=True,
    validate=True,
)

mean, std = compute_running_mean_std(train_raw, max_samples=512)

train_ds = SARDataset(
    img_root=img_root,
    mask_root=weak_mask_root,
    ids_or_paths=train_ids,
    mode="weak",
    normalize_cfg={"type": "zscore", "mean": mean, "std": std},
    log_transform=True,
)

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=default_collate,
)

images, masks, metas = next(iter(train_loader))
print(images.shape, masks.shape, metas[0]["id"])
```
