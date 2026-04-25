# `optical_dataset.py`

## Purpose

Implements `OpticalDataset`, the Sentinel-2 dataset for flood segmentation and optical-side multimodal pairing. It reads optical rasters, scales reflectance, clips outliers, handles missing/cloud pixels through `valid_mask`, optionally loads binary masks, and mirrors the `SARDataset` tuple contract.

## Why This File Exists

Optical imagery has different failure modes from SAR: exports may be DN-scaled or already reflectance-scaled, all-zero pixels commonly represent nodata, and SCL/cloud masks should exclude cloudy pixels from training. Keeping those rules in one dataset prevents model code from mixing optical preprocessing with training logic.

## Dependencies

Internal:

- Reuses `NormalizeCfg`, `Sample`, quantile cache helpers, and path/ID helpers from `sar_dataset.py`.

External:

- `warnings`, `os`, `time`
- `pathlib.Path`
- `numpy`
- `rasterio`
- `torch`
- `torch.utils.data.Dataset`

## Classes

### `OpticalDataset`

Responsibility: expose Sentinel-2 rasters as PyTorch samples with optional labels and pixel-validity metadata.

Constructor arguments:

| Argument | Meaning |
| --- | --- |
| `img_root` | S2 root when `ids_or_paths` are IDs. |
| `mask_root` | Label root for labeled modes. |
| `ids_or_paths` | Non-empty IDs or paths. |
| `mode` | `"weak"`, `"strong"`, or `"none"`. |
| `s2_bands` | Optional zero-based band subset. |
| `cloud_mask_root` | Optional root for cloud/SCL masks with matching stems. |
| `cloud_mask_invalid_values` | SCL values to mark invalid, default `(3,8,9,10,11)`. |
| `reflectance_clip_percentile` | Per-band clipping percentile, default `2.0`; `0.0` disables clipping. |
| `transforms` | Optional `(image, mask, metadata)` transform pipeline. |
| `normalize_cfg` | `"none"` or z-score config. |
| `validate` | Open files and drop invalid samples at init. |
| `ids_are_paths` | Optional ID/path inference override. |
| `mask_id_suffix_map` | Optional image-ID to mask-ID suffix mapping. |
| `expected_img_bands` | Expected channel count after band selection. |
| `quantile_cache_root` | Optional cache root for reflectance clipping quantiles. |

Internal state:

- Mode, band selection, cloud mask config, normalization config.
- Quantile cache/profile state.
- Resolved `samples`.

Public API:

- `__len__`
- `__getitem__`
- `__repr__`
- `normalize_type`
- `quantile_profile_summary`

`__len__` behavior: returns `len(self.samples)`.

`__getitem__` behavior:

1. Loads and preprocesses image via `_load_image`.
2. Verifies image is finite and valid mask is 2-D.
3. Converts image to `torch.float32`.
4. Converts `valid_mask` to `torch.bool`.
5. Loads binary mask in labeled modes.
6. Builds metadata with `modality="optical"`.
7. Applies transforms if configured.
8. Returns `(img_tensor, mask_tensor, metadata)`.

Returned Sample Schema:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `image` | `[C,H,W]` | `torch.float32` | S2 reflectance-derived tensor after band selection, scaling, clipping, imputation, optional normalization. |
| `mask` | `[1,H,W]` or `None` | `torch.uint8` | Binary flood label in labeled modes. |
| `valid_mask` | `[H,W]` | `torch.bool` | `True` where every selected band is finite/non-nodata/non-cloud. |
| `metadata` | `dict` | - | `id`, `img_path`, `mask_path`, `valid_mask`, `modality="optical"`. |

Invariants:

- Output image is finite.
- `valid_mask.ndim == 2` before transforms.
- Mask shape equals image spatial shape when present.
- If `expected_img_bands` is set, output channel count equals it.

Assumptions:

- Values whose 99th percentile exceeds `5.0` are DN-scaled and should be divided by `10000`.
- All-zero pixels across all selected bands are nodata.
- Cloud mask stems match image stems.

Failure behavior:

- Invalid mode, empty IDs, invalid clip percentile, missing required roots, or bad expected band counts raise at init.
- Missing cloud masks warn and skip cloud invalidation.
- Raster read errors become `RuntimeError`.
- Bad band indices raise `ValueError`.

## Functions

### `_parse_normalize_cfg(normalize_cfg)`

Purpose: validate and store normalization mode and z-score stats. Logic mirrors SAR.

### `_apply_normalization(img)`

Purpose: apply z-score normalization when configured.

Preconditions: stats length matches channel count and std values are positive.

### Quantile profiling helpers

`_record_quantile_shape`, `_record_nanquantile_time`, `_record_cache_hit`, `_record_cache_miss`, `_record_load_image_time`, and `quantile_profile_summary` mirror SAR profiling behavior.

### `_resolve_raster_path(root, name_or_id)`

Purpose: resolve an ID or TIFF name against a root, preferring existing `.tif`, then `.tiff`.

### `_resolve_mask_id(sample_id)`

Purpose: map image ID to mask ID, using direct existence first and suffix mapping second.

### `_build_samples(ids_or_paths)`

Purpose: build `Sample` dictionaries from IDs or paths.

### `_validate_samples(samples)`

Purpose: open images and masks at init and drop invalid samples with warnings.

### `_load_image(img_path, sample_id)`

Purpose: load and preprocess Sentinel-2 image; returns `(img, valid_mask)`.

Pipeline:

1. Read selected bands or all bands.
2. Check `expected_img_bands` after selection.
3. Convert to `float32`.
4. Detect DN scale using finite-value 99th percentile and divide by `10000` if needed.
5. Mark non-finite pixels invalid.
6. Mark all-zero pixels across all selected bands invalid.
7. Clamp values to `[0,1]`.
8. Optionally robust-clip per band using `reflectance_clip_percentile` and cache.
9. Optionally read cloud/SCL mask and mark configured values invalid.
10. Compute `valid_mask = ~invalid.any(axis=0)`.
11. Band-min impute remaining non-finite values.
12. Apply normalization.

Outputs:

- `img`: `np.float32`, `[C,H,W]`, finite.
- `valid_mask`: bool, `[H,W]`.

### `_load_mask(mask_path, expected_hw, sample_id)`

Purpose: read first mask band, check shape, binarize as `(mask > 0)`, return `[1,H,W]` `uint8`.

### `normalize_type`

Purpose: expose `_normalize_type` as a read-only property.

## Data Contracts

Modes mirror `SARDataset`: `"weak"` and `"strong"` require masks; `"none"` does not load masks.

Metadata keys:

| Key | Always | Meaning |
| --- | --- | --- |
| `id` | yes | Sample ID. |
| `img_path` | yes | Optical raster path. |
| `mask_path` | yes | Mask path or `None`. |
| `valid_mask` | yes | Pixel validity. |
| `modality` | yes | `"optical"`. |

## Tensor Shape Expectations

- Optical image: `[C,H,W]`, where `C=len(s2_bands)` when `s2_bands` is provided, otherwise raster band count.
- Mask: `[1,H,W]`.
- Valid mask: `[H,W]`.

## Metadata Structure

Metadata is intentionally close to SAR metadata so `default_collate`, `PatchDataset`, and `FusedDataset` can treat both modalities uniformly.

## Valid Mask Handling

`valid_mask=False` means the pixel is not safe for training due to non-finite values, all-zero nodata, or cloud/SCL invalidation. The image tensor is imputed to finite values after invalidity is recorded.

ENFORCED:

- `valid_mask` is two-dimensional before transforms.
- Image is finite after preprocessing.

ASSUMED:

- All-zero across selected bands is nodata. This may be wrong for legitimate very dark pixels if all selected reflectance bands are exactly zero.
- Cloud/SCL mask spatial shape matches image shape; the code broadcasts it without explicit shape check before invalid OR operation.

ACCIDENTAL:

- Missing cloud masks produce warnings and leave pixels valid instead of failing.

## Transform / Augmentation Behavior

Transforms run after image, optional mask, and metadata are created. They must preserve tuple schema. Optical-specific train transforms are usually applied outside the dataset in multimodal pretraining, not inside child datasets.

## Design Decisions

- DN/reflection auto-detection reduces caller burden when exports differ.
- Optical `log_transform` is intentionally absent.
- Robust clipping is configurable with `reflectance_clip_percentile`.
- Cloud invalidation is encoded in `valid_mask`, not in the label mask.

## Edge Cases and Failure Modes

- If finite values are empty, DN detection treats the image as not DN-scaled.
- If every value in a band is non-finite, imputation fill becomes `0.0`.
- Cached quantiles also record whether DN scaling was used.
- Out-of-range `s2_bands` fail at item load time.

## Modification Risks

- Changing DN threshold changes all optical normalization behavior.
- Treating cloud pixels as label ignore rather than `valid_mask` would require loss-code updates.
- Changing metadata keys can break `FusedDataset` and `PatchDataset`.

## Example Usage

```python
ds = OpticalDataset(
    img_root="S2Hand",
    mask_root="LabelHand",
    ids_or_paths=train_ids,
    mode="strong",
    s2_bands=[0, 1, 2, 3],
    cloud_mask_root="S2SCL",
    normalize_cfg="none",
)
```

## Open Questions / Ambiguities

- Whether cloud mask shape should be explicitly validated.
- Whether all-zero nodata should be configurable for datasets with valid zero reflectance.
- Whether DN threshold `5.0` is robust across all preprocessing exports.

