# `combined_manifest.py`

## Purpose

Loads and validates strict paired sample records from a `Combined/manifest.csv` dataset root.

## Why This File Exists

Directory globbing can silently pair the wrong SAR and optical tiles when IDs drift, files are missing, or subdirectories contain extra assets. This module makes the manifest the canonical source of membership, order, and asset paths for strict multimodal workflows.

## Dependencies

Internal: none.

External: `csv`, `dataclasses`, `pathlib`, `rasterio`.

## Classes

### `CombinedManifestSample`

Responsibility: immutable row-level record for one paired sample from a combined manifest.

Constructor arguments / fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | `str` | Canonical ID from manifest. |
| `manifest_index` | `int` | Zero-based row index in manifest iteration. |
| `manifest_path` | `str` | Path to the manifest file used. |
| `sar_path` | `str` | Resolved existing SAR file path. |
| `optical_path` | `str` | Resolved existing optical file path. |
| `label_path` | `Optional[str]` | Resolved label path when present or required. |

Internal state: dataclass fields only; frozen after construction.

Public API: attribute access.

`__len__` behavior: none.

`__getitem__` behavior: none.

Invariants:

- When created by `load_combined_manifest_samples`, `sar_path` and `optical_path` exist.
- `sample_id` equals the file stem of resolved manifest assets.

## Functions

### `load_combined_manifest_samples(combined_root_or_manifest, require_label=False)`

Purpose: parse `manifest.csv`, resolve asset paths, validate required row fields, and return ordered `CombinedManifestSample` objects.

Inputs:

- `combined_root_or_manifest`: combined root directory or path to `manifest.csv`.
- `require_label`: when true, `output_Label` is required and must resolve.

Outputs: `list[CombinedManifestSample]` in manifest row order.

Side effects: opens the manifest file and checks filesystem existence for referenced assets.

Preconditions:

- Combined root exists.
- Manifest exists and has a header.
- Required columns are present.
- Asset cell values are relative paths under expected subdirectories.

Postconditions:

- Returned list is non-empty.
- Every sample has SAR and optical paths.
- Label paths are present when required and optional otherwise.

Where it is used: `FusedDataset.from_combined_manifest`.

Why it exists: makes strict multimodal pairing explicit and auditable.

Edge cases:

- Empty manifest raises `ValueError`.
- Missing `sample_id` raises with row index.
- Path outside expected directory raises even if the file exists elsewhere.

### `validate_combined_manifest_samples(samples, require_spatial_match=True, require_label=False)`

Purpose: assert that referenced rasters are readable and jointly shape-compatible.

Inputs:

- `samples`: sequence of `CombinedManifestSample`.
- `require_spatial_match`: require SAR and optical height/width equality.
- `require_label`: require labels and check SAR/label spatial equality.

Outputs: `None`.

Side effects: opens raster files through rasterio.

Preconditions: samples contain paths to rasterio-readable assets.

Postconditions: raises on unreadable files, zero-band files, spatial mismatch, or missing required labels.

Where it is used: `FusedDataset.from_combined_manifest(validate=True)`.

Why it exists: separates manifest structural parsing from raster-level integrity checks.

Edge cases: validation catches dimensions only, not CRS/geotransform alignment.

### `_resolve_manifest_inputs(combined_root_or_manifest)`

Purpose: normalize caller input into `(combined_root, manifest_path)`.

Inputs: root directory or manifest path.

Outputs: tuple of `Path`.

Side effects: filesystem existence checks.

### `_require_manifest_columns(fieldnames, manifest_path, extra_required=())`

Purpose: enforce manifest header columns.

Inputs: CSV field names, manifest path for error reporting, optional extra required names.

Outputs: `None`.

Side effects: none.

### `_resolve_manifest_asset(combined_root, raw_value, expected_dir, sample_id, column, manifest_index)`

Purpose: resolve and validate one asset cell.

Inputs: combined root, raw CSV cell, expected first path component, sample ID, column name, row index.

Outputs: existing asset path as `str`.

Side effects: filesystem existence check.

Postconditions:

- Returned path is inside `expected_dir`.
- File stem equals `sample_id`.
- File exists.

### `_read_raster_shape(path, sample_id, kind)`

Purpose: read `(height,width)` from a raster and reject zero-band assets.

Inputs: path and diagnostic labels.

Outputs: `(height, width)`.

Side effects: opens raster file.

## Data Contracts

Manifest rows must use:

| Column | Required | Expected value |
| --- | --- | --- |
| `sample_id` | always | Non-empty canonical file stem. |
| `output_S1` | always | Relative path under `S1/` with stem equal to `sample_id`. |
| `output_S2` | always | Relative path under `S2/` with stem equal to `sample_id`. |
| `output_Label` | when `require_label=True`; optional otherwise | Relative path under `Label/` with stem equal to `sample_id`. |

## Tensor Shape Expectations

This module does not create tensors. It validates raster spatial dimensions as `(height,width)`.

## Metadata Structure

`CombinedManifestSample` fields are later copied into fused metadata: `manifest_path`, `manifest_row_index`, `label_path`, and paired asset paths via SAR/optical child metadata.

## Valid Mask Handling

None directly. Strict manifest records feed child datasets that create valid masks.

## Transform / Augmentation Behavior

None.

## Design Decisions

ENFORCED:

- Manifest row order is preserved exactly.
- Asset subdirectory and stem are checked.
- Missing files fail before dataset construction in strict mode.

ASSUMED:

- Equal raster dimensions are enough for strict paired training unless external preprocessing handled georegistration.
- `sample_id` is globally unique within the manifest. Duplicate sample IDs are not explicitly checked here.

ACCIDENTAL:

- Extra manifest columns are accepted and ignored.

## Edge Cases and Failure Modes

- Passing a path named `manifest.csv` makes its parent the combined root.
- Passing a directory assumes `directory/manifest.csv`.
- Duplicate `sample_id` rows are not rejected here.
- `rasterio.open` exceptions are wrapped as `ValueError` by `_read_raster_shape`.

## Modification Risks

- Relaxing stem or directory checks weakens strict-pairing guarantees.
- Changing row order invalidates reproducibility and random splits in multimodal pretraining.
- Adding fallback path resolution would make manifest errors harder to diagnose.

## Example Usage

```python
samples = load_combined_manifest_samples("datasets/FilteredSouthAsia/Combined")
validate_combined_manifest_samples(samples, require_spatial_match=True)
```

## Open Questions / Ambiguities

- CRS, affine transform, pixel alignment, and acquisition-date matching are not validated.
- Duplicate `sample_id` policy is not specified.

