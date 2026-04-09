# Data Loader

`src/data_loader` contains the dataset stack used for SAR, optical, fused, and
patch-based training.

## Core datasets

`SARDataset`
- Returns `(image, mask, metadata)`.
- Supports `mode="strong"`, `"weak"`, or `"none"`.
- `metadata` includes `id`, `img_path`, `mask_path`, and `valid_mask`.

`OpticalDataset`
- Mirrors the same dataset contract for Sentinel-2 tiles.
- Supports band selection, optional cloud masking, and the same `valid_mask`
  semantics.

`FusedDataset`
- Concatenates SAR and optical channels in `[C_sar + C_s2, H, W]`.
- Returns the SAR mask when labels are present.
- Legacy direct construction still pairs by sample ID and may use
  `optical_missing_policy="zeros"` for ad hoc experiments.

## Strict paired fusion

For multimodal unlabeled pretraining, use:

```python
from pathlib import Path

from data_loader import FusedDataset

fused_ds = FusedDataset.from_combined_manifest(
    Path("datasets/FilteredSouthAsia/Combined"),
    mode="none",
    sar_dataset_kwargs={
        "normalize_cfg": "none",
        "log_transform": True,
    },
    optical_dataset_kwargs={
        "s2_bands": [0, 1, 2, 3],
        "normalize_cfg": "none",
    },
)
```

Strict paired mode guarantees:
- Membership comes from `Combined/manifest.csv`, not directory globbing.
- Sample order matches manifest row order exactly.
- Every sample has one SAR tile and one optical tile with the same canonical
  `sample_id`.
- Missing optical fallback is not allowed.
- Validation is joint/assertive rather than silently dropping one modality.
- Metadata preserves canonical provenance (`id`, `paired_sample_id`,
  `sar_img_path`, `optical_img_path`, `label_path`, `manifest_row_index`,
  `strict_paired_mode`).

For pre-augmentation paired inspection, construct strict fused data with child
`transforms=None` and use `get_paired_item(..., require_no_transforms=True)`:

```python
pair = fused_ds.get_paired_item(0, require_no_transforms=True)
sar = pair["sar_image"]
optical = pair["optical_image"]
metadata = pair["metadata"]
```

You can also save a quick visualization without patching:

```bash
python scripts/visualize_sen12ms_pair.py path/to/Combined --index 0 --output pair.png
```

Do not treat raw `S1Hand` / `S2Hand` directory pairing as equivalent to strict
paired mode. Those roots are useful for legacy or modality-specific workflows,
but strict multimodal fusion should be built from `Combined`.

## Patching

Wrap any base dataset with `PatchDataset` to create deterministic overlapping
patches while preserving provenance in metadata (`base_id`, `patch_row`,
`patch_col`, `patch_y0`, `patch_x0`, `patch_size`).

## Utilities

- `load_combined_manifest_samples(...)`: ordered canonical strict-pair manifest
  reader.
- `make_split(ids, val_frac, seed)`: deterministic split helper.
- `compute_running_mean_std(dataset, max_samples=None)`: streaming stats.
- `default_collate(...)`: collate function that keeps metadata aligned.
- `validate_sample_shapes(ds, n=64)`: quick shape/dtype sanity checks.
