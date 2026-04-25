# `collate.py`

## Purpose

Provides DataLoader collate functions for the two active sample schemas: segmentation tuple samples `(image, mask, metadata)` and paired multimodal pretraining dict samples `{"sar", "optical", "valid_mask", "meta"}`.

## Why This File Exists

PyTorch's default collation cannot preserve this repository's metadata and validity-mask semantics cleanly. These functions batch tensors while keeping metadata aligned with sample order and while enforcing important invariants before tensors reach the model.

## Dependencies

Internal: none.

External: `torch` and typing helpers.

## Classes

No classes are defined.

## Functions

### `default_collate(batch)`

Purpose: collate segmentation-style samples returned by `SARDataset`, `OpticalDataset`, `FusedDataset(return_mode="fused")`, and `PatchDataset` wrappers over those datasets.

Inputs:

- `batch`: sequence of `(image, mask, metadata)` tuples.

Outputs:

| Output | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `images` | `[B,C,H,W]` | usually `float32` | Stacked image tensors. |
| `masks` | `[B,1,H,W]` or `None` | usually `uint8` | Stacked masks, or `None` if all samples are unlabeled. |
| `valid_masks` | `[B,H,W]` | bool-like | Pixel validity masks extracted from metadata. |
| `metas` | `list[dict]` | - | Metadata shallow copies with `valid_mask` removed. |

Side effects: shallow-copies metadata dictionaries and does not mutate original samples.

Preconditions:

- `batch` is not empty.
- All images have identical shape.
- Masks are either all `None` or all tensors with identical shape.
- `metadata["valid_mask"]`, if present, is `[H,W]` or `[1,H,W]` and stack-compatible.

Postconditions:

- `valid_mask` is removed from returned metadata dicts and returned as a dedicated tensor.
- Missing `valid_mask` is replaced by an all-true mask matching image spatial shape.

Where it is used: example scripts and intended supervised SAR, optical, and fused segmentation DataLoaders.

Why it exists: keeps pixel-validity masks in a dense tensor and rejects mixed labeled/unlabeled batches.

Edge cases:

- Empty batches raise `ValueError`.
- Mixed `None` and tensor masks raise `ValueError`.
- The function does not force `valid_masks.bool()` after stacking.

### `multimodal_pretrain_collate(batch)`

Purpose: collate strict paired multimodal pretraining samples into a dict batch consumed by DeCUR training/evaluation.

Inputs:

- `batch`: sequence of dictionaries containing `sar`, `optical`, `valid_mask`, and `meta`.

Outputs:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `sar` | `[B,C_sar,H,W]` | `float32` | Stacked SAR tensors. |
| `optical` | `[B,C_opt,H,W]` | `float32` | Stacked optical tensors. |
| `valid_mask` | `[B,1,H,W]` | `bool` | Stacked joint validity masks. |
| `meta` | `list[dict]` | - | Shallow-copied paired metadata. |

Side effects: none beyond local tensor coercion.

Preconditions:

- Every sample is a dict.
- Required keys are present.
- `meta` is a dict.
- `sar` and `optical` are `[C,H,W]`.
- SAR and optical spatial dimensions match within each sample.
- Every SAR tensor in the batch has the same full shape.
- Every optical tensor in the batch has the same full shape.
- `valid_mask` is `[H,W]` or `[1,H,W]`.
- `meta` does not contain `mask_path` or `label_path`.

Postconditions:

- Output tensors are stack-compatible and typed for training.
- Valid mask has a channel dimension `[B,1,H,W]`.

Where it is used:

- `src/Multi_modal_src/train.py` passes this to `DataLoader`.
- `src/Multi_modal_src/eval.py` expects this exact batch schema.

Why it exists: pretraining is unlabeled and paired by modality, so keeping SAR and optical separate avoids hiding modality identity inside concatenated channels and avoids label leakage.

Edge cases:

- Empty batch raises `ValueError`.
- Label-bearing metadata raises `ValueError`.
- Shape mismatch between SAR and optical raises before the model sees the batch.

## Data Contracts

`default_collate` consumes tuple samples. `multimodal_pretrain_collate` consumes paired dict samples. They are not interchangeable.

ENFORCED:

- Empty batch rejection.
- Mixed labeled/unlabeled rejection in tuple batches.
- Required keys and shape checks in paired batches.
- Label metadata rejection in paired batches.

ASSUMED:

- Tuple metadata may contain arbitrary non-tensor provenance after `valid_mask` removal.
- `valid_mask=True` means a pixel can be used by downstream training.

ACCIDENTAL:

- `default_collate` accepts non-bool valid masks and stacks them without final dtype coercion.

## Tensor Shape Expectations

Tuple path:

- Image `[C,H,W]` -> `[B,C,H,W]`.
- Mask `[1,H,W]` -> `[B,1,H,W]`.
- Valid mask `[H,W]` or `[1,H,W]` -> `[B,H,W]`.

Paired path:

- SAR `[C_sar,H,W]` -> `[B,C_sar,H,W]`.
- Optical `[C_opt,H,W]` -> `[B,C_opt,H,W]`.
- Valid mask `[H,W]` or `[1,H,W]` -> `[B,1,H,W]`.

## Metadata Structure

Tuple collate returns metadata without `valid_mask`. Paired collate returns metadata as a `meta` list and forbids `mask_path` and `label_path`.

## Valid Mask Handling

Tuple path extracts `valid_mask` from metadata and defaults missing masks to all true. Paired path requires `valid_mask`.

## Transform / Augmentation Behavior

No transforms are applied here. Collate expects datasets or training code to have already applied or scheduled transforms.

## Design Decisions

- Valid masks are batched separately for supervised segmentation because losses need dense tensors.
- Paired pretraining keeps SAR and optical separate because DeCUR consumes modality-specific views.
- Paired collate rejects label metadata to reduce accidental supervised leakage.

## Edge Cases and Failure Modes

- Variable image sizes cannot be collated by these functions; use `PatchDataset` or resizing before batching.
- If metadata contains nested mutable objects, shallow copies do not deeply isolate them.

## Modification Risks

- Changing paired `valid_mask` from `[B,1,H,W]` breaks `src/Multi_modal_src/eval.py`.
- Returning `valid_mask` inside tuple metadata as well as a tensor creates two sources of truth.
- Allowing mixed labeled/unlabeled tuple batches complicates supervised loss code.

## Example Usage

```python
loader = DataLoader(dataset, batch_size=4, collate_fn=default_collate)
images, masks, valid_masks, metas = next(iter(loader))
```

```python
loader = DataLoader(fused_paired_dataset, batch_size=128, collate_fn=multimodal_pretrain_collate)
batch = next(iter(loader))
```

## Open Questions / Ambiguities

- Whether `default_collate` should force `valid_masks = valid_masks.bool()` is not specified.
- Whether `ignore_mask` should be extracted and batched like `valid_mask` is unresolved.

