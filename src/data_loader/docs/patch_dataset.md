# `patch_dataset.py`

## Purpose

Implements `PatchDataset`, a deterministic overlapping patch wrapper for full-tile datasets. It turns each base tile into many square samples while preserving image, mask, valid-mask, ignore-mask, and provenance alignment.

## Why This File Exists

Hand-labeled flood data is small. Patching increases the effective number of supervised samples and gives models fixed-size inputs without changing base dataset logic.

## Dependencies

Internal: accepts any dataset following the tuple contract.

External:

- `math`, `warnings`
- `collections.OrderedDict`
- `pathlib.Path`
- `numpy` (imported but not used directly in current code)
- `rasterio`
- `torch`
- `torch.utils.data.Dataset`

## Classes

### `_PatchIndex`

Responsibility: lightweight container for one patch coordinate.

Constructor arguments:

| Field | Meaning |
| --- | --- |
| `base_idx` | Index into wrapped dataset. |
| `row` | Patch grid row. |
| `col` | Patch grid column. |
| `y0` | Top pixel coordinate. |
| `x0` | Left pixel coordinate. |

Internal state: fixed `__slots__` fields.

Public API: attribute access only.

`__len__` behavior: none.

`__getitem__` behavior: none.

### `PatchDataset`

Responsibility: expose deterministic patches from a base dataset returning `(image, mask, metadata)`.

Constructor arguments:

| Argument | Meaning |
| --- | --- |
| `base_dataset` | Dataset to patch. |
| `patch_size` | Square patch size in pixels. |
| `overlap` | Fractional overlap in `[0,1)`. |
| `skip_mostly_nodata` | Stored but not currently implemented in `_build_index`. |
| `nodata_threshold` | Stored but not currently implemented. |
| `processed_tile_cache_size` | LRU cache size for processed full tiles when base has no transforms. |

Internal state:

- `stride = max(1, ceil(patch_size * (1-overlap)))`
- `_index: list[_PatchIndex]`
- optional processed tile LRU cache keyed by base index

Public API:

- `__len__`
- `__getitem__`
- `patches_per_image_stats`
- `__repr__`

`__len__` behavior: returns number of generated patch index entries.

`__getitem__` behavior:

1. Gets `_PatchIndex`.
2. Loads base item, optionally from cache.
3. Crops image `[C,H,W]` to `[C,patch_size,patch_size]`.
4. Crops mask, if present.
5. Shallow-copies metadata.
6. Rewrites `id` to patch ID and adds patch provenance.
7. Crops `valid_mask` and `ignore_mask` if present.
8. Validates cropped mask spatial alignment.
9. Returns `(img_patch, mask_patch, patch_meta)`.

Returned Sample Schema:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `image` | `[C,patch_size,patch_size]` normally | same as base | Cropped image patch. |
| `mask` | `[1,patch_size,patch_size]` or `None` normally | same as base | Cropped label patch. |
| `valid_mask` | `[patch_size,patch_size]` or `[1,patch_size,patch_size]` | bool-like | Cropped validity mask. |
| `metadata` | `dict` | - | Base metadata plus patch coordinates and base ID. |

Invariants:

- Patch image, mask, valid mask, and ignore mask use the same crop coordinates.
- Patch metadata includes enough information to map back to the source tile.

Assumptions:

- Base dataset returns channel-first image tensors.
- Base dataset metadata has an image path discoverable through `samples`, `sar_dataset`, nested `Subset`, or metadata fallback.
- Base image dimensions read from raster path match tensor dimensions returned by base dataset.

Failure behavior:

- Invalid overlap or patch size raises at init.
- No generated patches raises `ValueError`.
- Bad mask/valid-mask/ignore-mask ranks or spatial mismatch raises during `__getitem__`.

## Functions

### `_grid_starts(length, patch, stride)`

Purpose: generate top-left coordinates that cover a 1-D axis.

Inputs: axis length, patch size, stride.

Outputs: list of start indices.

Postconditions:

- If `patch >= length`, returns `[0]`.
- Otherwise includes a final start at `length - patch` so the boundary is covered.

### `_tile_shape(img_path)`

Purpose: read `(height,width)` from raster metadata without reading pixels.

Side effects: opens raster file.

### `_dataset_has_transforms(dataset)`

Purpose: recursively detect whether a dataset or nested `dataset` child has transforms enabled.

Why it exists: caching processed full tiles is unsafe when base transforms are random.

### `_dataset_image_path(dataset, idx)`

Purpose: find the image path for a base item without loading pixels when possible.

Lookup order:

1. Nested `Subset`-style `dataset`/`indices`.
2. `samples[idx]["img_path"]`.
3. `sar_dataset` child for fused datasets.
4. `None`.

### `_build_index()`

Purpose: enumerate all patch coordinates for all base samples using raster metadata.

Side effects: opens raster files and warns/skips samples whose shape cannot be read.

### `_get_base_item(base_idx)`

Purpose: return full processed base item, using LRU cache if caching is enabled.

Caching rule: enabled only when `processed_tile_cache_size > 0` and no transforms are detected on base dataset.

### `patches_per_image_stats()`

Purpose: return diagnostic counts over generated patch index.

Outputs:

- `n_base_images`
- `n_patches_total`
- `mean_patches_per_image`
- `min_patches_per_image`
- `max_patches_per_image`

## Data Contracts

Base dataset must return:

| Position | Expected |
| --- | --- |
| `image` | tensor `[C,H,W]` |
| `mask` | tensor `[1,H,W]` or `None` |
| `metadata` | dict, ideally with `id`, `img_path`, and `valid_mask` |

Patch metadata additions:

| Key | Meaning |
| --- | --- |
| `id` | `<base_id>_r<row>_c<col>` |
| `base_id` | Original metadata ID. |
| `patch_row` | Grid row. |
| `patch_col` | Grid column. |
| `patch_y0` | Source top coordinate. |
| `patch_x0` | Source left coordinate. |
| `patch_size` | Configured patch size. |

## Tensor Shape Expectations

Expected patch shape is `[*, patch_size, patch_size]`. If source raster is smaller than patch size, Python slicing returns the smaller source extent even though `_grid_starts` returns `[0]`; this is a contract risk.

## Metadata Structure

Metadata is shallow-copied from base metadata. Tensor masks are cloned when cropped, but nested non-tensor metadata is not deep-copied.

## Valid Mask Handling

`valid_mask` is cropped if present and rank is `[H,W]` or `[1,H,W]`. The shape is checked against the image patch.

ENFORCED:

- Valid-mask rank must be 2 or singleton-channel 3.
- Cropped valid-mask spatial shape must match image patch.

ASSUMED:

- Missing valid-mask is acceptable and will be handled later by `default_collate` as all true.

ACCIDENTAL:

- `skip_mostly_nodata` and `nodata_threshold` do not currently filter patches.

## Transform / Augmentation Behavior

`PatchDataset` itself does not apply transforms. It caches base samples only when no transforms are detected to avoid reusing random augmentations across patches. If transforms are applied inside the base dataset before patching, those transforms affect full tiles before crops.

## Design Decisions

- Patch index is built from raster metadata at init without loading image pixels.
- Edge-aligned final starts guarantee coverage of full raster extent.
- Patching is a wrapper so base datasets keep ownership of reading, preprocessing, and labels.

## Edge Cases and Failure Modes

- Wrapped `random_split` datasets are supported through `dataset`/`indices` traversal.
- Fused datasets are supported by looking through `sar_dataset` for image paths.
- If a base item lacks discoverable path, index building may load the item and inspect metadata.
- Very small rasters can yield smaller-than-configured patches.

## Modification Risks

- Implementing nodata filtering requires loading masks during index build, which changes init cost significantly.
- Caching transformed base tiles would make random augmentation deterministic per cached tile.
- Changing patch ID format can break downstream provenance parsing.

## Example Usage

```python
patch_ds = PatchDataset(base_ds, patch_size=256, overlap=0.2)
image, mask, meta = patch_ds[0]
```

## Open Questions / Ambiguities

- Whether `skip_mostly_nodata` should be implemented or removed.
- Whether small rasters should be padded to `patch_size` or rejected.
- Whether patch filtering should use labels, valid masks, or image nodata.

