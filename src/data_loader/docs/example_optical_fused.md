# `example_optical_fused.py`

## Purpose

Provides examples for optical-only training, strict SAR/optical fused loading from `Combined/manifest.csv`, and legacy non-strict fusion with missing-optical fallback.

## Why This File Exists

It documents the intended use of `OpticalDataset`, `FusedDataset`, `PatchDataset`, stats computation, validation, and collate functions in realistic workflows.

## Dependencies

Internal:

- `FusedDataset`
- `OpticalDataset`
- `PatchDataset`
- `SARDataset`
- `compute_running_mean_std`
- `default_collate`
- `make_split`
- `validate_sample_shapes`

External:

- `pathlib.Path`
- `torch.utils.data.DataLoader`

## Classes

No classes are defined.

## Functions

### `optical_only_example()`

Purpose: demonstrate supervised optical-only dataset construction.

Inputs: uses module-level roots and IDs.

Outputs: prints stats and batch shapes.

Side effects: reads data, computes stats, validates samples, creates patches and DataLoader batch.

Preconditions: module-level paths and IDs must exist.

Postconditions: demonstrates `default_collate` output.

Where it is used: script entrypoint example.

Why it exists: shows S2 band selection, cloud masking, z-score stats, and patching.

### `strict_fused_example()`

Purpose: demonstrate strict manifest-backed fusion.

Inputs: `COMBINED_ROOT`.

Outputs: prints dataset repr and batch shape diagnostics.

Side effects:

- Loads/validates combined manifest.
- Computes separate SAR and optical stats from child datasets.
- Constructs normalized strict fused dataset.
- Validates and patches it.

Preconditions: combined root contains valid `manifest.csv` and assets.

Postconditions: shows fused images and strict metadata fields.

Why it exists: documents the recommended strict path for multimodal data.

### `fused_missing_optical_example()`

Purpose: demonstrate legacy ad hoc fusion where optical data may be absent.

Inputs: SAR and partial S2 roots/IDs.

Outputs: prints `fused_optical_available`, image shape, and mask status.

Side effects: constructs non-strict `FusedDataset(optical_missing_policy="zeros")`.

Why it exists: explicitly labels this path as non-strict and legacy-tolerant.

## Data Contracts

Optical-only and fused examples use tuple samples and `default_collate`.

Strict fused example uses `FusedDataset.from_combined_manifest(..., return_mode` default `"fused"`), not paired DeCUR mode.

## Tensor Shape Expectations

- Optical-only patched batch: `[B,C_s2,patch_size,patch_size]`.
- Fused patched batch: `[B,C_sar+C_s2,patch_size,patch_size]`.

## Metadata Structure

Examples inspect:

- `id`
- `n_sar_bands`
- `n_optical_bands`
- `strict_paired_mode`
- `fused_optical_available`

## Valid Mask Handling

Both optical and fused examples collate `valid_masks` as `[B,H,W]`; they do not show the loss application in this file.

## Transform / Augmentation Behavior

No augmentation transforms are configured directly. Patching happens after dataset preprocessing.

## Design Decisions

- Computes modality-specific stats from child datasets instead of from concatenated fused tensors.
- Uses strict combined manifest for the recommended paired workflow.
- Keeps legacy missing-optical example separate and warns that it is not strict paired training.

## Edge Cases and Failure Modes

- Module-level IDs are placeholders and likely do not exist.
- `masks` may be `None` in strict fused example because it uses `mode="none"`.
- Validation only checks first few samples in example calls.

## Modification Risks

- Treating legacy zero-fill fusion as equivalent to strict pairing can silently break multimodal experiments.
- Computing normalization on fused concatenated tensors would mix SAR and optical statistics.

## Example Usage

```bash
python -m data_loader.example_optical_fused
```

## Open Questions / Ambiguities

- Whether an additional example should show `return_mode="paired"` for `Multi_modal_src/train.py`.

