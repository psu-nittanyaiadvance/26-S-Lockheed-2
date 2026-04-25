# `stats.py`

## Purpose

Computes per-band running mean and standard deviation for dataset normalization.

## Why This File Exists

Large raster datasets should not be loaded fully into memory just to compute normalization statistics. This module uses a streaming Welford-style update and excludes invalid pixels.

## Dependencies

Internal: expects the dataset tuple contract but imports no loader modules.

External: `numpy`, `torch`, `torch.utils.data.Dataset`.

## Classes

No classes are defined.

## Functions

### `compute_running_mean_std(dataset, max_samples=None, require_no_transforms=False)`

Purpose: compute per-channel mean and standard deviation over valid finite pixels.

Inputs:

- `dataset`: object implementing `__len__` and `__getitem__`, returning `(image, mask, metadata)`.
- `max_samples`: optional cap on sampled items from the beginning of the dataset.
- `require_no_transforms`: if true, reject datasets whose `transforms` attribute is not `None`.

Outputs: `(mean, std)` as `np.float32` arrays of shape `[C]`.

Side effects: iterates through dataset samples and triggers raster reads, preprocessing, warnings, and any enabled transforms.

Preconditions:

- Dataset is non-empty.
- Each sampled item returns image `[C,H,W]`.
- Metadata `valid_mask`, if present, is compatible with image spatial shape.

Postconditions:

- Only pixels where `valid_mask` is true and band value is finite are included.
- Raises if no valid pixels exist or if any band has zero valid pixels.

Where it is used: example scripts compute raw stats before constructing z-score-normalized datasets.

Why it exists: keeps normalization robust to NaN/Inf/nodata pixels and avoids memory blowup.

Edge cases:

- If metadata lacks `valid_mask`, it falls back to `np.isfinite(arr).all(axis=0)`.
- If image is a `torch.Tensor`, it is detached and moved to CPU; valid-mask tensors are converted with `vm.numpy()` without explicit CPU/detach.

## Data Contracts

Dataset item contract:

| Position | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `image` | `[C,H,W]` | numeric | Image values after current dataset preprocessing. |
| `_mask` | any | ignored | Label not used for statistics. |
| `metadata["valid_mask"]` | `[H,W]` | bool-like | Pixels included in statistics when true. |

ENFORCED:

- Empty dataset rejected.
- Image must be 3-D.
- Bands with zero valid pixels rejected.

ASSUMED:

- Use `normalize_cfg="none"` and `transforms=None` to compute raw stats.
- Valid masks are CPU tensors or NumPy arrays.

ACCIDENTAL:

- `require_no_transforms` checks only `getattr(dataset, "transforms", None)`, so wrappers with child transforms may pass unless they expose that attribute.

## Tensor Shape Expectations

Input image `[C,H,W]`; output stats `[C]`.

## Metadata Structure

Uses only `metadata["valid_mask"]`.

## Valid Mask Handling

Valid mask is treated as cross-band pixel inclusion. A pixel must be globally valid and the current band value must be finite.

## Transform / Augmentation Behavior

No transforms are applied directly, but dataset-level transforms will run if enabled unless `require_no_transforms=True` catches them.

## Design Decisions

- Welford parallel update avoids storing all pixels.
- Cross-band `valid_mask` prevents imputed invalid pixels from biasing stats.
- Per-band counts allow reporting bands with no usable data.

## Edge Cases and Failure Modes

- `max_samples=0` on a non-empty dataset results in no processed pixels and raises.
- Metadata valid-mask shape is not explicitly checked before boolean indexing; incompatible shape will raise from NumPy indexing.

## Modification Risks

- Including invalid imputed pixels would shift normalization, especially for SAR band-min imputation.
- Computing stats after z-score normalization would return near-zero/one but would not be useful for initial config.

## Example Usage

```python
raw = SARDataset(..., normalize_cfg="none", transforms=None)
mean, std = compute_running_mean_std(raw, max_samples=512, require_no_transforms=True)
```

## Open Questions / Ambiguities

- Whether random subsampling should be supported instead of taking the first `max_samples`.
- Whether valid masks should be detached and moved to CPU defensively like images.

