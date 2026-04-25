# `fused_dataset.py`

## Purpose

Implements `FusedDataset`, the SAR/optical pairing layer. It can return either a segmentation-style fused tensor with SAR and optical channels concatenated, or a paired multimodal dictionary with SAR and optical kept separate for DeCUR pretraining.

## Why This File Exists

SAR and optical datasets have different preprocessing but multimodal models need aligned samples. This file owns pairing, strict manifest enforcement, valid-mask composition, and modality provenance so those concerns do not leak into model code.

## Dependencies

Internal:

- `combined_manifest.CombinedManifestSample`
- `combined_manifest.load_combined_manifest_samples`
- `combined_manifest.validate_combined_manifest_samples`
- `optical_dataset.OpticalDataset`
- `sar_dataset.SARDataset`

External:

- `pathlib.Path`
- `torch`
- `torch.utils.data.Dataset`

## Classes

### `FusedDataset`

Responsibility: align SAR and optical samples and expose either fused-channel or paired-modality samples.

Constructor arguments:

| Argument | Meaning |
| --- | --- |
| `sar_dataset` | Dataset returning SAR-style `(image, mask, metadata)` samples. |
| `optical_dataset` | Dataset returning optical-style `(image, mask, metadata)` samples. |
| `optical_missing_policy` | `"raise"`, `"zeros"`, or deprecated `"sar_only"` alias to `"zeros"`. |
| `require_spatial_match` | Whether SAR and optical spatial dimensions must match. |
| `strict_pairing` | Whether manifest-backed exact pairing invariants are enforced. |
| `strict_manifest_samples` | Manifest records used in strict mode. |
| `return_mode` | `"fused"` for tuple samples, `"paired"` for dict pretraining samples. |

Internal state:

- Child datasets.
- Optical ID-to-index lookup.
- Inferred optical band count for zero-fill fallback.
- Strict-pairing flag and manifest records.
- Return mode.

Public API:

- `from_combined_manifest`
- `__len__`
- `__getitem__`
- `get_paired_item`
- `__repr__`

`__len__` behavior: returns `len(self.sar_dataset)`. SAR membership is the base membership in both strict and non-strict modes.

`__getitem__` behavior:

- If `return_mode="paired"`, delegates to `_paired_training_item`.
- If `return_mode="fused"`, loads pair, composes valid masks, concatenates SAR and optical channels, builds fused metadata, and returns `(fused_img, mask_tensor, metadata)`.

Returned Sample Schema for `return_mode="fused"`:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `image` | `[C_sar+C_opt,H,W]` | `torch.float32` | SAR channels followed by optical channels. |
| `mask` | `[1,H,W]` or `None` | `torch.uint8` | SAR-side mask from `sar_dataset`. |
| `valid_mask` | `[H,W]` | `torch.bool` | `sar_valid & optical_valid` when optical exists, else SAR valid mask. |
| `metadata` | `dict` | - | SAR metadata plus fused pairing fields. |

Returned Sample Schema for `return_mode="paired"`:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `sar` | `[C_sar,H,W]` | `torch.float32` | SAR tensor. |
| `optical` | `[C_opt,H,W]` | `torch.float32` | Optical tensor. |
| `valid_mask` | `[H,W]` | `torch.bool` | `sar_valid & optical_valid`. |
| `meta` | `dict` | - | Label-free paired provenance. |

Invariants:

- Strict mode has exact SAR, optical, and manifest ID order parity.
- `return_mode="paired"` requires optical data for every sample.
- `valid_mask` spatial shape matches image shape.
- In fused mode, SAR channels always precede optical channels.

Assumptions:

- Matching IDs imply aligned modalities in non-strict mode.
- Strict manifest creation handled geospatial/time pairing beyond file shape.

Failure behavior:

- Missing optical raises under `optical_missing_policy="raise"` and under paired return mode.
- Strict mode rejects non-raise missing policy.
- Spatial mismatch raises when `require_spatial_match=True`.
- Bad valid-mask shape raises with sample/source diagnostics.

## Functions

### `from_combined_manifest(combined_root, mode="none", sar_dataset_kwargs=None, optical_dataset_kwargs=None, require_spatial_match=True, validate=True, return_mode="fused")`

Purpose: construct a strict manifest-backed `FusedDataset`.

Inputs:

- Combined root or manifest path.
- Dataset mode; labels are required only when `mode != "none"`.
- Child dataset kwargs, excluding ownership fields managed by strict mode.
- Validation and return-mode flags.

Outputs: strict `FusedDataset`.

Side effects:

- Reads manifest.
- Optionally opens rasters for validation.

Preconditions:

- Manifest columns and assets satisfy `combined_manifest.py` contracts.
- Caller does not pass forbidden child kwargs (`ids_or_paths`, `img_root`, `mask_root`, `ids_are_paths`, `mode`, or `validate`).

Postconditions:

- Child datasets use manifest paths in manifest order.
- SAR masks are patched from manifest label paths in labeled modes.
- `strict_pairing=True` and `optical_missing_policy="raise"`.

Where it is used:

- `src/Multi_modal_src/train.py` uses this with `mode="none"` and `return_mode="paired"`.

Why it exists:

- Provides one safe entry point for strict multimodal data.

Edge cases:

- If `combined_root` is a manifest file path, its parent becomes combined root.

### `_paired_training_item(idx)`

Purpose: return separate SAR and optical tensors for contrastive multimodal training.

Preconditions:

- Optical tile must exist.
- Spatial dimensions must match when required.

Postconditions:

- Metadata excludes `mask_path` and `label_path`, satisfying `multimodal_pretrain_collate`.

### `get_paired_item(idx, require_no_transforms=False)`

Purpose: inspect one paired sample with SAR and optical tensors kept separate, including modality-specific metadata and valid masks.

Inputs:

- `idx`
- `require_no_transforms`: reject if child dataset transforms are enabled.

Outputs:

| Key | Meaning |
| --- | --- |
| `paired_sample_id` | ID used for both modalities. |
| `sar_image` | SAR tensor. |
| `optical_image` | Optical tensor. |
| `mask` | SAR-side mask. |
| `sar_metadata` | SAR metadata with coerced `valid_mask`. |
| `optical_metadata` | Optical metadata with coerced `valid_mask`. |
| `metadata` | Combined inspection metadata with `sar_valid_mask` and `optical_valid_mask`. |

Why it exists: supports visualization/debugging before or after augmentation decisions.

### `_load_pair(idx)`

Purpose: load SAR sample and matching optical sample or zero-fill optical fallback.

Outputs: internal dict containing paired ID, tensors, metadata, mask, and availability flag.

Edge cases:

- In zero-fill mode, optical metadata uses SAR valid mask and `img_path=None`.

### `_assert_no_child_transforms()`

Purpose: reject paired inspection when child transforms are enabled.

### `_reject_strict_dataset_overrides(kwargs, dataset_name)`

Purpose: prevent callers from overriding strict manifest-managed child dataset fields.

### `_assert_strict_pairing()`

Purpose: assert manifest, SAR child dataset, and optical child dataset have exact ID and path parity.

### `_ordered_dataset_ids(dataset)`

Purpose: obtain dataset IDs from `samples` when available or by loading items as fallback.

### `_infer_optical_bands()`

Purpose: infer optical channel count once for zero-fill fallback.

Edge cases: returns `1` if optical dataset is empty or all samples fail to load.

### `_coerce_valid_mask(valid_mask, expected_hw, sample_id, source)`

Purpose: normalize valid masks to bool `[H,W]` and validate shape.

### `__repr__()`

Purpose: compact diagnostic string with SAR count, optical count, paired count, policy, and strict flag.

## Data Contracts

Fused metadata keys include:

| Key | Meaning |
| --- | --- |
| `id` | Paired sample ID. |
| `img_path` | SAR image path for backward compatibility. |
| `sar_img_path` | SAR image path. |
| `optical_img_path` | Optical image path or `None`. |
| `mask_path` | SAR mask path. |
| `valid_mask` | Joint valid mask. |
| `modality` | `"fused"` or `"paired_multimodal"`. |
| `fused_optical_available` | Whether real optical data was present. |
| `n_sar_bands` | SAR channel count. |
| `n_optical_bands` | Optical channel count. |
| `paired_sample_id` | Canonical pair ID. |
| `strict_paired_mode` | Strict mode flag. |
| `pairing_source` | `"combined_manifest"` or `"id_lookup"`. |
| `manifest_path` | Strict mode only. |
| `manifest_row_index` | Strict mode only. |
| `label_path` | Strict fused mode only. |

## Tensor Shape Expectations

- SAR image: `[C_sar,H,W]`
- Optical image: `[C_opt,H,W]`
- Fused image: `[C_sar+C_opt,H,W]`
- Mask: `[1,H,W]` or `None`
- Valid mask: `[H,W]`

## Metadata Structure

Fused mode starts from SAR metadata and overwrites/adds fused fields. Paired mode builds a new metadata dict intentionally free of label-bearing path fields.

## Valid Mask Handling

SAR and optical valid masks are coerced independently. When both modalities are available, joint validity is logical AND. In zero-fill fallback, optical validity is treated as SAR validity because optical data is artificial.

ENFORCED:

- Valid mask rank and shape.
- Joint valid-mask composition in both return modes.

ASSUMED:

- Logical AND is the correct multimodal validity rule; a pixel invalid in either modality should not drive paired training.

ACCIDENTAL:

- Zero-filled optical fallback marks optical availability false but keeps pixels valid where SAR is valid, which can silently train a fused segmentation model with fake optical channels if the user chooses `zeros`.

## Transform / Augmentation Behavior

`FusedDataset` itself does not apply transforms. Child datasets may have transforms, but strict multimodal pretraining in `Multi_modal_src/train.py` passes transforms later in `build_decur_batch_views`. `get_paired_item(require_no_transforms=True)` can enforce unaugmented inspection.

## Design Decisions

- SAR membership drives dataset length.
- SAR mask owns segmentation labels in fused mode.
- Strict paired mode is manifest-backed and disallows missing optical fallback.
- `return_mode="paired"` exists because contrastive pretraining should not concatenate modalities.

## Edge Cases and Failure Modes

- Duplicate optical IDs in non-strict mode overwrite earlier indices in `_optical_id_to_idx`.
- `_infer_optical_bands` can hide optical loading failures by returning `1`, affecting zero-fill fallback shape.
- `require_spatial_match=False` allows concatenation of tensors with different spatial sizes only until `torch.cat`, which will fail unless sizes still match.

## Modification Risks

- Changing channel order breaks any model expecting SAR-first fused tensors.
- Adding label paths to paired metadata breaks `multimodal_pretrain_collate`.
- Relaxing strict-pairing assertions can silently corrupt multimodal training.
- Changing valid-mask composition from AND changes contrastive view masking semantics.

## Example Usage

```python
fused = FusedDataset.from_combined_manifest(
    "datasets/FilteredSouthAsia/Combined",
    mode="none",
    return_mode="paired",
    sar_dataset_kwargs={"expected_img_bands": 2},
    optical_dataset_kwargs={"expected_img_bands": 13},
)
sample = fused[0]
```

## Open Questions / Ambiguities

- Whether strict validation should compare CRS/transform/grid, not only dimensions.
- Whether non-strict duplicate IDs should be fatal.
- Whether zero-fill fallback should be allowed for anything except explicit legacy experiments.

