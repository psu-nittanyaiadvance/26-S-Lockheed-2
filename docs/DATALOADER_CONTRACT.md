# DataLoader Contract (Training Loop Integration)

## 1) Purpose and Scope
This document defines the strict, testable interface contract between the data loader and the training loop. It is the single source of truth for tensor shapes, dtypes, metadata keys, valid enum values, and failure modes. The goal is to allow independent development without cross-editing code.

Contract version: 1.2.0 (2026-02-20)

## 2) Terminology (ID, split, mode, time-matched, weak/strong, labeled/unlabeled)
ID: Filename stem of a SAR GeoTIFF (e.g., `tile_000123` from `tile_000123.tif`). IDs are the canonical sample identifiers.
Split: Dataset partition token inferred from path or provided by the caller. For time-matched stacks, the loader infers `split_inferred` from image path tokens.
Mode: Label regime. Valid values: `weak`, `strong`, `none`.
Time-matched: An 8-band SAR+Optical stack aligned in space/time with the SAR tile, stored as a GeoTIFF.
Weak/strong: Weak labels are auto-generated; strong labels are hand-labeled. The loader does not map weak to strong IDs.
Labeled/unlabeled: Labeled means a mask is present (`mode` is `weak` or `strong`). Unlabeled means `mode` is `none`.

## 3) Supported Dataset Layouts (paths + examples)
Primary layout (repo convention):
- Weak images: `datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak`
- Weak masks: `datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak`
- Strong images: `datasets/FilteredSouthAsia/HandLabeled/S1Hand`
- Strong masks: `datasets/FilteredSouthAsia/HandLabeled/LabelHand`

Time-matched artifacts:
- Stacks: `data/derived/gee_time_matched/<split>/<id>.tif`
- Manifest: `data/derived/gee_time_matched_manifest.csv`

ID resolution:
- If IDs are provided: image is resolved to `img_root/<id>.tif` (fallback to `.tiff`).
- If paths are provided: image path is used directly; mask resolves to `mask_root/<stem>.tif`.

Example:
- ID input: `ids_or_paths=["tile_000123"]` with `img_root=".../S1Weak"` loads `.../S1Weak/tile_000123.tif`.
- Path input: `ids_or_paths=["D:/data/S1Weak/tile_000123.tif"]` uses that exact file.

## 4) Inputs (constructor args / config fields) with types and allowed values
`SARDataset` constructor (from `src/data_loader/sar_dataset.py`):
- `img_root`: `str | Path | None`. Required when IDs are used.
- `mask_root`: `str | Path | None`. Required for `mode` in `{weak, strong}`.
- `ids_or_paths`: `Sequence[str | Path]`. Non-empty. IDs or paths (auto-inferred).
- `mode`: `str`. Allowed: `weak`, `strong`, `none`.
- `transforms`: `Callable | None`. Called as `transforms(image, mask, metadata)`.
- `normalize_cfg`: `None | "none" | {"type": "zscore", "mean": [...], "std": [...]}`.
- `log_transform`: `bool`. If `True`, uses `log1p(abs(x))`.
- `validate`: `bool`. If `True`, drops samples with unreadable rasters or mismatched sizes.
- `ids_are_paths`: `bool | None`. If `None`, inferred from inputs.
- `use_time_matched`: `bool`.
- `time_matched_root`: `str | Path`. Default `data/derived/gee_time_matched`.
- `time_matched_manifest`: `str | Path`. Default `data/derived/gee_time_matched_manifest.csv`.
- `time_matched_missing_policy`: `str`. Allowed: `zeros`, `skip`, `raise`.
- `split_tokens`: `Sequence[str]`. Tokens to infer split from paths. Must be non-empty.
- `expected_img_bands`: `int | None`. If set, SAR band count must match.
- `expected_time_matched_bands`: `int`. Default `8`. Must be positive.

## 5) Outputs per __getitem__
Returned structure:
- Tuple: `(image, mask, metadata)`

`image`:
- Type: `torch.Tensor`
- Dtype: `torch.float32`
- Shape: `[C, H, W]`
- Range: arbitrary floats. If `log_transform=True`, values are `log1p(abs(x))` of original.

`mask`:
- Type: `torch.Tensor | None`
- Dtype: `torch.uint8` when present
- Shape: `[1, H, W]` when present
- Values: binary `{0, 1}`
- `None` when `mode="none"`

`metadata` (dict):
Required keys always present:
- `id`: `str` (sample ID)
- `img_path`: `str` (resolved image path)
- `mask_path`: `str | None` (resolved mask path, `None` if `mode="none"`)

Standard keys for labeled modes (`mode in {"weak","strong"}`):
- `ignore_mask`: `torch.Tensor` with shape `[1, H, W]` and dtype `torch.bool`. Pixels marked `True` should be excluded from loss/metrics. This is always present for labeled samples; if there are no ignored pixels it is an all-false tensor.

Required keys when `use_time_matched=True`:
- `time_matched`: `torch.Tensor` with shape `[8, H, W]` and dtype `torch.float32` when available, or a zeros tensor if missing policy is `zeros`
- `time_matched_path`: `str | None`
- `time_matched_status`: `str | None` (e.g., `ok`, `missing`, `missing_s1`, `missing_s2`, `missing_both`, `error`)
- `split_inferred`: `str` (split token inferred from image path, or `all`)

Integration-required fields:
- `mode` and `has_mask` are required for the training loop but are not stored in `metadata`. The training loop must track `mode` from dataset config and derive `has_mask` as `mask is not None`.

## 6) Mask Semantics
Label encoding (strong labels on disk):
- Raw masks use `-1` for ignore, `0` for background, `1` for flood.
- Masks are binarized on load using `mask > 0` and output as `uint8`.
Ignore index:
- This contract does not encode ignore values in the mask tensor. Instead `metadata["ignore_mask"]` (shape `[1,H,W]`, `bool`) marks ignored pixels and is always present for labeled modes (all-false when no ignores).
When masks may be missing:
- `mode="none"`: masks are always `None`.
- `mode in {"weak","strong"}`: masks must exist; missing masks raise at access.

## 7) Time-Matched Stack Contract
Shape and band count:
- Expected shape: `[8, H, W]`.
- Expected band order for time-matched stacks: `[VV, VH, B2, B3, B4, B8, B11, B12]`.
- If `expected_time_matched_bands` is set (default `8`) and a stack has a different band count, `ValueError` is raised.

Manifest vs filesystem:
- If `time_matched_manifest` exists, it is loaded and indexed by `sample_id`.
- For each sample:
  - If manifest status is `ok` and the path exists, that path is used.
  - Otherwise, it falls back to `time_matched_root/<split>/<id>.tif` where `split` is inferred.
- If manifest is missing, a warning is emitted and filesystem checks are used.

Missing policy:
- `zeros`: return an all-zero `[8, H, W]` stack and emit `TIME_MATCHED_ZEROS_WARNING` once per dataset instance.
- `skip`: drop samples missing a valid stack during dataset initialization; if none remain, `ValueError` is raised.
- `raise`: raise `RuntimeError` at `__getitem__` if a time-matched stack is missing.

## 8) Transform + Normalization Rules
Ordering:
- Load SAR image.
- Apply `log_transform` if enabled: `log1p(abs(x))`.
- Apply normalization if `normalize_cfg` is `zscore`.
- Convert to `torch.float32`.

Normalization:
- `normalize_cfg="none"` or `None` disables normalization.
- `normalize_cfg={"type":"zscore","mean":[...],"std":[...]}` applies per-band z-score.
- If mean/std lengths do not match band count, or any std <= 0, `ValueError` is raised.

Stats (Welford) constraints:
- `compute_running_mean_std` must fail if:
  - `dataset.transforms` is not `None` and `require_no_transforms=True`.
  - `dataset` is `SARDataset` and normalization is enabled with `require_normalize_none=True`.
- Log-transform consistency is enforced if `include_log_transform_flag` is set.

## 9) Collate Contract
`default_collate`:
- Inputs: batch of `(image, mask, metadata)` tuples.
- Outputs: `(images, masks, metas)`.
- Shapes:
  - `images`: `[B, C, H, W]`
  - `masks`: `[B, 1, H, W]` or `None`
  - `metas`: list of dicts, length `B`
- Failure mode: raises `ValueError` if batch mixes labeled and unlabeled samples.

## 10) Validation and Smoke Tests
Compliance must be verifiable using:
- `validate_sample_shapes(ds, n=...)`
  - Enforces `image` dtype `float32` and finite values.
  - Enforces `mask` dtype `uint8` and binary `{0,1}`.
- `validate_time_matched(ds, n=...)`
  - Enforces time-matched shape/dtype and missing-policy behavior.
- `validate_manifest_consistency(path)`
  - Enforces manifest schema and status values.
- `scripts/smoke_test_loader.py`
  - End-to-end dataset + collate + validation.
- `scripts/smoke_test_time_matched.py`
  - Ensures time-matched loading and missing policy.

Tests that must align with this contract:
- `tests/test_data_utils.py`
  - Time-matched zeros warning emitted once.
  - Stats guardrails reject transforms.
  - Band-count checks for SAR and time-matched stacks.

## 11) Training Loop Integration
Expected batch structure:
- From `default_collate`: `(images, masks, metas)` where `masks` may be `None`.
 - Mixed labeled/unlabeled batches are not supported; use separate loaders for `mode="none"` vs labeled modes.

Loss behavior:
- For Tversky loss, compute supervised loss only on labeled batches. For inference-only data, use `mode="none"` and skip supervised loss.

Minimal integration example (pseudocode only):
```text
dataset = SARDataset(..., mode=MODE, use_time_matched=USE_TM, normalize_cfg=NORM, ...)
collate = default_collate
loader = DataLoader(dataset, batch_size=B, collate_fn=collate, ...)

for images, masks, metas in loader:
    if masks is None:
        skip supervised loss or run inference-only path
    else:
        loss = criterion(preds, masks)  # labeled batches only
    log: mode, roots, split ids, missing policy, channel counts
```

## 12) Error Handling and Invariants
Invariants (must always hold):
- `image` is `torch.float32` and 3D `[C,H,W]`.
- `mask` is `torch.uint8` and `[1,H,W]` when present.
- `mask` values are binary `{0,1}` before collate.
- `time_matched` is `torch.float32` and `[8,H,W]` when enabled.
- `TIME_MATCHED_ZEROS_WARNING` is emitted only once per dataset instance when policy is `zeros`.

Errors and warnings (expected behavior):
- `ValueError` if `ids_or_paths` is empty.
- `ValueError` if `mode` not in `{weak, strong, none}`.
- `ValueError` if `img_root` or `mask_root` is missing or invalid when required.
- `ValueError` if `time_matched_missing_policy` not in `{zeros, skip, raise}`.
- `ValueError` if expected band counts are violated.
- `RuntimeError` if raster read fails for image or mask.
- Warning if time-matched manifest not found (fallback to filesystem).
- Warning if split inference falls back to `all` for many samples.
- `ValueError` if `default_collate` sees mixed labeled/unlabeled samples.

All failure modes above are testable via `validate_dataset`, `tests/test_data_utils.py`, and smoke tests.

## 13) Reporting Requirements for Ablations
Every run must log the following fields:
- Dataset roots (image root, mask root, time-matched root).
- Split seed and exact train/val ID file paths (or a deterministic description).
- Mode (`weak`, `strong`, `none`).
- `use_time_matched` and missing policy (`zeros`, `skip`, `raise`).
- Normalization config source (mean/std file or computed on train split).
- Channel counts (SAR-only = 2, SAR + time-matched = 8).

## 14) Versioning + Breaking Changes
Versioning:
- Use semantic versioning for this contract document.
- Increment:
  - MAJOR for breaking interface changes.
  - MINOR for backwards-compatible additions.
  - PATCH for clarifications or doc fixes.

Breaking change checklist:
- Any change to tensor shapes, dtypes, or value ranges.
- Any change to `metadata` required keys or meaning.
- Any change to `mode` or missing-policy enum values.
- Any change to time-matched band order or count.
- Any change to mask encoding or binary semantics.
- Any change to collate return structure or error behavior.
- Any change to required logging fields for ablations.
- Any change that invalidates `tests/test_data_utils.py` or `validate_dataset` expectations.
