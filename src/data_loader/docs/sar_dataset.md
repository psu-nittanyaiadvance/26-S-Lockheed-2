# `sar_dataset.py`

## Purpose

Implements `SARDataset`, the Sentinel-1 dataset for flood segmentation and SAR-side multimodal pairing. It reads SAR rasters, creates finite model tensors, creates pixel-validity masks, optionally loads binary flood labels, and can attach time-matched auxiliary stacks in metadata.

## Why This File Exists

SAR data needs modality-specific preprocessing. Sentinel-1 backscatter can contain non-finite values, non-positive values that break log transforms, and outliers. The dataset centralizes those rules so training code receives finite tensors plus an explicit `valid_mask` telling losses which pixels are trustworthy.

## Dependencies

Internal: none for primary dataset logic.

External:

- `csv`, `hashlib`, `os`, `time`, `warnings`
- `pathlib.Path`
- `numpy`
- `torch`
- `torch.utils.data.Dataset`
- `rasterio`
- `rasterio.errors.RasterioIOError`

## Classes

### `Sample`

Responsibility: `TypedDict` describing one dataset row.

Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `str` | Sample ID, normally image stem. |
| `img_path` | `str` | Raster path for SAR image. |
| `mask_path` | `Optional[str]` | Raster path for label mask in labeled modes. |

### `SARDataset`

Responsibility: expose Sentinel-1 rasters as PyTorch samples with optional labels and valid-mask metadata.

Constructor arguments:

| Argument | Meaning |
| --- | --- |
| `img_root` | Image root used when `ids_or_paths` are IDs. Required unless paths are provided. |
| `mask_root` | Mask root for `mode="weak"` or `"strong"`. |
| `ids_or_paths` | Non-empty list of IDs, TIFF names, or full paths. |
| `mode` | `"weak"`, `"strong"`, or `"none"`. Controls mask loading only. |
| `transforms` | Optional callable accepting `(image, mask, metadata)`. |
| `normalize_cfg` | `None`, `"none"`, or `{"type":"zscore","mean":...,"std":...}`. |
| `log_transform` | Whether to apply `np.log` after flagging non-positive values. |
| `validate` | Whether to open files at init and drop invalid samples. |
| `ids_are_paths` | Optional override for ID/path inference. |
| `use_time_matched` | Whether to attach an 8-band time-matched stack in metadata. |
| `time_matched_root` | Root for fallback time-matched files. |
| `time_matched_manifest` | CSV manifest for time-matched stacks. |
| `time_matched_missing_policy` | `"zeros"`, `"skip"`, or `"raise"`. |
| `mask_id_suffix_map` | Optional suffix mapping from image IDs to mask IDs. |
| `expected_img_bands` | Optional hard check on raster band count. |
| `quantile_cache_root` | Optional cache root for robust clipping quantiles. |

Internal state:

- `mode`, `transforms`, `log_transform`
- path roots and ID/path behavior
- optional time-matched index
- normalization type and stats
- quantile cache settings and profiling counters
- `samples: list[Sample]`

Public API:

- `__len__`
- `__getitem__`
- `quantile_profile_summary`

`__len__` behavior: returns `len(self.samples)`.

`__getitem__` behavior:

1. Reads and preprocesses image via `_load_image`.
2. Converts image to `float32` tensor.
3. Converts valid mask to bool tensor.
4. Loads mask in labeled modes.
5. Builds metadata with paths and `valid_mask`.
6. Optionally attaches time-matched tensor metadata.
7. Applies transforms if configured.
8. Returns `(img_tensor, mask_tensor, metadata)`.

Returned Sample Schema:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `image` | `[C,H,W]` | `torch.float32` | SAR image after clipping, optional log transform, imputation, and optional normalization. |
| `mask` | `[1,H,W]` or `None` | `torch.uint8` | Binary flood label in labeled modes; absent in `mode="none"`. |
| `valid_mask` | `[H,W]` | `torch.bool` | `True` where every SAR band was valid through preprocessing. |
| `metadata` | `dict` | - | Contains `id`, `img_path`, `mask_path`, `valid_mask`, optional `ignore_mask`, optional time-matched fields. |

Invariants:

- `image` is finite after `_load_image`.
- `valid_mask=False` covers non-finite input and log-invalid pixels.
- `mask.shape[-2:] == image.shape[-2:]` when mask exists.
- `ignore_mask`, when present, is bool and spatially aligned with mask.

Assumptions:

- `mask > 0` means flood/positive label.
- File stem is the sample ID unless caller supplies suffix mapping for masks.
- Quantile clipping is appropriate for all SAR bands.

Failure behavior:

- Invalid mode, empty IDs, missing roots, bad expected band counts, bad normalization configs, or bad time-matched policy raise during initialization.
- Raster read errors become `RuntimeError`.
- Mask/image shape mismatch raises `ValueError`.
- Missing time-matched stack raises only under `time_matched_missing_policy="raise"`.

## Functions

### `_is_tif_name(name)`

Purpose: detect `.tif`/`.tiff` names. Used by path and ID resolution.

### `_infer_ids_are_paths(items)`

Purpose: infer whether `ids_or_paths` contains paths based on absolute paths, separators, or TIFF suffixes.

ASSUMED: any separator means path-like; this may classify nested relative IDs as paths.

### `_infer_split_from_path(path)`

Purpose: return `"WeaklyLabeled"`, `"HandLabeled"`, or `"all"` from path parts for time-matched fallback lookup.

### `_quantile_cache_path(cache_root, dataset_type, sample_id, img_path)`

Purpose: build a stable cache path using sanitized sample ID and resolved file-path digest.

### `_file_cache_fingerprint(img_path)`

Purpose: create a cache key fragment from resolved path, mtime, and size.

### `_load_quantile_cache(cache_path, expected_key)`

Purpose: load cached lower/upper tensors if the key matches.

Outputs: cached dict or `None`.

Failure behavior: returns `None` on missing/corrupt/incompatible cache.

### `_save_quantile_cache(cache_path, key, lower, upper, extra=None)`

Purpose: atomically write quantile cache payload.

Side effects: creates directories and writes `.pt` file; warns on failure.

### `_load_time_matched_index()`

Purpose: read `time_matched_manifest` into a dictionary keyed by `sample_id`.

Side effects: warns if manifest is absent.

### `_time_matched_path(sample)`

Purpose: resolve a time-matched stack path from manifest first, then filesystem fallback.

### `_time_matched_exists(sample)`

Purpose: decide whether a sample has an acceptable time-matched stack for skip filtering.

### `_parse_normalize_cfg(normalize_cfg)`

Purpose: validate and store normalization mode and z-score stats.

### `_resolve_raster_path(root, name_or_id)`

Purpose: resolve an ID or TIFF name against a root, preferring existing `.tif`, then `.tiff`, and returning `.tif` fallback.

### `_resolve_mask_id(sample_id)`

Purpose: map an image ID to a mask ID, preferring direct mask existence before suffix mapping.

### `_build_samples(ids_or_paths)`

Purpose: create `Sample` records from IDs or paths.

### `_validate_samples(samples)`

Purpose: open rasters and masks at init, warn/drop bad samples, and warn on high invalid/non-positive SAR fractions.

Postcondition: returns only valid samples.

### `_apply_normalization(img)`

Purpose: apply configured normalization.

Preconditions: z-score stats length must match channel count and stds must be positive.

### Quantile profiling helpers

`_record_quantile_shape`, `_record_nanquantile_time`, `_record_cache_hit`, `_record_cache_miss`, `_record_load_image_time`, and `quantile_profile_summary` track optional performance counters when `LOCKDOCKS_PROFILE_QUANTILES=1`.

### `_load_image(img_path, sample_id)`

Purpose: load and preprocess SAR image; returns `(img, valid_mask)`.

Pipeline:

1. Read raster `[C,H,W]`.
2. Check `expected_img_bands` if configured.
3. Robust-clip each band to 2nd/98th percentile using cache when possible.
4. Mark non-finite pixels invalid.
5. If `log_transform=True`, mark non-positive pixels invalid, set them to NaN, then log.
6. Compose `valid_mask = ~invalid.any(axis=0)`.
7. Band-min impute non-finite values to keep image tensor finite.
8. Apply normalization.

Outputs:

- `img`: `np.float32`, `[C,H,W]`, finite.
- `valid_mask`: bool, `[H,W]`.

### `_load_mask(mask_path, expected_hw, sample_id)`

Purpose: read first mask band, check shape, binarize as `(mask > 0)`, and return `[1,H,W]` `uint8`.

## Data Contracts

Modes:

| Mode | Image | Mask |
| --- | --- | --- |
| `"weak"` | required | required from `mask_root` |
| `"strong"` | required | required from `mask_root` |
| `"none"` | required | not loaded |

Metadata keys:

| Key | Always | Meaning |
| --- | --- | --- |
| `id` | yes | Sample ID. |
| `img_path` | yes | SAR raster path. |
| `mask_path` | yes | Mask path or `None`. |
| `valid_mask` | yes | Pixel validity. |
| `ignore_mask` | no | Optional weak-label ignore region. |
| `time_matched` | no | 8-band auxiliary stack tensor. |
| `time_matched_path` | if `use_time_matched` | Auxiliary stack path or `None`. |
| `time_matched_status` | if `use_time_matched` | Status string. |

## Tensor Shape Expectations

- SAR image: `[C,H,W]`
- Mask: `[1,H,W]`
- Valid mask: `[H,W]`
- Time-matched stack: `[8,H,W]`

## Metadata Structure

Metadata is a mutable dictionary passed to transforms. Transforms may add or mutate `ignore_mask` and spatially transform `valid_mask`.

## Valid Mask Handling

`valid_mask` is created from invalid input state, not from label state. It is `False` if any SAR band is invalid. The image tensor is still imputed to finite values so convolutions remain safe, but losses should exclude invalid pixels.

ENFORCED:

- Non-finite and log-invalid pixels are recorded before imputation.
- `_load_mask` checks mask spatial shape.

ASSUMED:

- Training code uses `valid_mask` in the loss.
- Per-band minimum imputation is acceptable because invalid pixels are excluded.

ACCIDENTAL:

- `valid_mask` does not include label uncertainty unless a transform adds `ignore_mask`; the two masks have separate semantics.

## Transform / Augmentation Behavior

If `transforms` is configured, it is called after image/mask/metadata construction. It may return `(image, mask, metadata)` or legacy `(image, mask)`. If `ignore_mask` existed before transforms and is dropped, `SARDataset` raises.

## Design Decisions

- Robust clipping happens before log transform, with quantiles cached by file fingerprint.
- Non-positive values are explicitly invalidated before log transform.
- Missing time-matched stacks can be zero-filled, skipped, or raised based on policy.
- `mode` affects label loading only; it does not change image preprocessing.

## Edge Cases and Failure Modes

- If every value in a band is non-finite, imputation fill becomes `0.0`.
- `validate=True` drops bad samples with warnings rather than failing immediately, then fails if none remain.
- `_resolve_raster_path` can return a non-existing `.tif` fallback; actual failure occurs on read or validation.
- `from numpy.strings import lower` appears unused.
- Encoding artifacts in comments/docstrings do not affect runtime.

## Modification Risks

- Changing log-transform invalidation changes the meaning of `valid_mask`.
- Moving normalization before imputation can reintroduce NaN/Inf.
- Changing mask binarization affects label semantics globally.
- Returning time-matched stacks outside metadata would break the tuple contract.

## Example Usage

```python
ds = SARDataset(
    img_root="S1Hand",
    mask_root="LabelHand",
    ids_or_paths=train_ids,
    mode="strong",
    normalize_cfg="none",
    log_transform=True,
)
image, mask, meta = ds[0]
```

## Open Questions / Ambiguities

- Whether quantile clipping should be optional/configurable beyond cache environment flags.
- Whether CRS/geotransform validation should be added for paired workflows.
- Whether high non-positive fractions should be fatal in production.

