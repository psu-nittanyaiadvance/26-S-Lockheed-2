# Data Loader (src/data_loader)

This folder contains the SAR-only dataset and utilities used by the baseline.
All samples are ground-truth masks (strong or weak labels).

**Core Dataset: `SARDataset`**

`SARDataset` returns `(image, mask, metadata)`:
- `image`: `torch.float32` with shape `[C, H, W]` read via rasterio.
- `mask`: `torch.uint8` with shape `[1, H, W]` (binary).
- `metadata`: dict with `id`, `img_path`, `mask_path`, and `ignore_mask`.

Notes:
- `mode` must be `strong` or `weak` (masks are always required).
- If `ids_are_paths=False`, pass stems in `ids_or_paths` and set `img_root`.
- If `ids_are_paths=True`, pass `.tif/.tiff` paths and set `mask_root`.
- `mask_id_suffix_map` can map image IDs to mask IDs when filenames differ.
- `normalize_cfg` supports `"none"` or `{"type": "zscore", "mean": ..., "std": ...}`.

**Patching**

Use `PatchDataset` (or `SARDataset.with_patches`) to generate overlapping
256×256 patches with 20% overlap (stride computed in `PatchDataset`).

**Utilities**

- `list_ids_from_dir(img_root)`: sorted stems for `.tif/.tiff` files.
- `paired_ids(img_root, mask_root)`: intersection of image/mask IDs.
- `make_split(ids, val_frac, seed)`: deterministic split helper.
- `compute_running_mean_std(dataset, max_samples=None)`: streaming stats.
- `validate_sample_shapes(ds, n=64)`: checks image/mask shapes and dtypes.

**Usage Example**

```python
from pathlib import Path
from torch.utils.data import DataLoader

from src.data_loader import (
    PatchDataset,
    SARDataset,
    compute_running_mean_std,
    default_collate,
    make_split,
)

img_root = Path("datasets/FilteredSouthAsia/HandLabeled/S1Hand")
mask_root = Path("datasets/FilteredSouthAsia/HandLabeled/LabelHand")

ids = ["tile_000123", "tile_000124", "tile_000125", "tile_000126"]
train_ids, val_ids = make_split(ids, val_frac=0.2, seed=1337)

train_raw = SARDataset(
    img_root=img_root,
    mask_root=mask_root,
    ids_or_paths=train_ids,
    mode="strong",
    normalize_cfg="none",
    log_transform=True,
)

mean, std = compute_running_mean_std(train_raw, max_samples=512)

train_ds = SARDataset(
    img_root=img_root,
    mask_root=mask_root,
    ids_or_paths=train_ids,
    mode="strong",
    normalize_cfg={"type": "zscore", "mean": mean, "std": std},
    log_transform=True,
)

patch_ds = PatchDataset(train_ds, patch_size=256, overlap=0.2)

train_loader = DataLoader(
    patch_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=default_collate,
)

images, masks, metas = next(iter(train_loader))
print(images.shape, masks.shape, metas[0]["id"])
```
