# `validate_dataset.py`

## Purpose

Provides quick runtime validation of sample shapes, dtypes, finiteness, masks, and selected modality-specific assumptions.

## Why This File Exists

Dataset construction may succeed even when a later model or collate path would fail. This helper samples the first `n` items and catches common data contract violations early.

## Dependencies

Internal: relies on metadata conventions from dataset classes but imports none.

External: `numpy`, `torch`.

## Classes

No classes are defined.

## Functions

### `_as_numpy(arr)`

Purpose: convert tensors or array-like objects to NumPy arrays.

Inputs: any array-like object.

Outputs: `np.ndarray`.

Side effects: tensor inputs are detached and moved to CPU.

### `_require(condition, message)`

Purpose: raise `ValueError` with a message when a condition fails.

Inputs: boolean and message.

Outputs: `None`.

### `_iter_indices(n, total)`

Purpose: produce `range(min(n,total))`.

Inputs: requested count and total dataset length.

Outputs: iterable indices.

### `_detect_modality(meta)`

Purpose: infer modality string from metadata.

Inputs: metadata dict.

Outputs: `"fused"` if `fused_optical_available` is present; otherwise `meta.get("modality", "sar")`.

Design rationale: preserve backwards compatibility for SAR datasets that do not set `metadata["modality"]`.

### `validate_sample_shapes(ds, n=64)`

Purpose: validate first `n` samples of `SARDataset`, `OpticalDataset`, or `FusedDataset`.

Inputs:

- `ds`: dataset object.
- `n`: maximum number of leading samples to inspect.

Outputs: `None`.

Side effects: calls `ds[idx]`, which loads rasters and applies dataset transforms.

Preconditions:

- Dataset implements `__len__` and `__getitem__`.
- Items use segmentation tuple schema.

Postconditions: raises `ValueError` if a sample violates expected image or mask contract.

Where it is used: `example_optical_fused.py`.

Why it exists: provides a low-friction smoke test before training.

Edge cases:

- Does not validate paired dict samples from `FusedDataset(return_mode="paired")`.
- Only checks the first `n` samples.

### `_validate_optical_image(img_np, sample_id, ds)`

Purpose: check optical `[0,1]` range when normalization type is `"none"` or unknown.

Inputs: image array, sample ID, and dataset or fused-like object.

Outputs: `None`.

Side effects: none.

Preconditions: image has already been checked as `float32` and finite.

## Data Contracts

Validated image contract:

| Field | Expected |
| --- | --- |
| image rank | 3 |
| image dtype | `np.float32` |
| image values | finite |
| mask shape | `[1,H,W]` |
| mask dtype | `np.uint8` |
| mask values | subset of `{0,1}` |

Fused metadata contract:

- If `n_sar_bands` and `n_optical_bands` are present, their sum must equal image channels.

## Tensor Shape Expectations

Segmentation tuple samples only: image `[C,H,W]`, mask `[1,H,W]`.

## Metadata Structure

Reads `id`, `modality`, `fused_optical_available`, `n_sar_bands`, and `n_optical_bands`.

## Valid Mask Handling

This helper does not validate `valid_mask`, despite valid-mask importance elsewhere.

## Transform / Augmentation Behavior

If the dataset has transforms, validation sees transformed samples. This may be useful for end-to-end checking but can make random failures nondeterministic.

## Design Decisions

ENFORCED:

- Image dtype, rank, and finiteness.
- Mask dtype/rank/binary values.
- Fused channel-count consistency when metadata exposes counts.

ASSUMED:

- Optical data without z-score normalization should be in `[0,1]`.
- Missing modality metadata means SAR.

ACCIDENTAL:

- `fused_optical_available` presence defines modality as fused even if value is `False`.
- Valid-mask shape is not checked here.

## Edge Cases and Failure Modes

- `n=0` on non-empty dataset performs no per-sample checks and returns.
- Datasets returning paired dicts will fail unpacking.
- Z-score optical range is intentionally not constrained.

## Modification Risks

- Strengthening optical range checks could reject valid z-score data if normalize-type inference is wrong.
- Adding valid-mask checks would be useful but could expose existing shape drift.

## Example Usage

```python
validate_sample_shapes(fused_ds, n=4)
```

## Open Questions / Ambiguities

- Whether valid-mask validation should be added.
- Whether paired multimodal dict validation should be a separate helper.

