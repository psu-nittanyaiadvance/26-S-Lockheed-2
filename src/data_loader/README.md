# Data Loading Subsystem

This directory owns the repository's flood-segmentation and multimodal pretraining data contracts. It turns raster files and manifests into PyTorch samples, carries pixel-validity information through preprocessing and augmentation, and provides collate functions that keep images, labels, validity masks, and provenance aligned.

The loader is intentionally conservative: missing files, mismatched paired assets, non-finite pixels, wrong channel counts, and mixed labeled/unlabeled batches are either rejected or recorded explicitly. The main exception is legacy fused loading with `optical_missing_policy="zeros"`, which keeps a SAR sample usable by zero-filling optical channels.

## Overview

The subsystem supports four dataset views:

- `SARDataset`: Sentinel-1 image tiles with optional flood masks.
- `OpticalDataset`: Sentinel-2 image tiles with optional flood masks and optional cloud/SCL masking.
- `FusedDataset`: SAR and optical samples paired by ID or strict `Combined/manifest.csv` rows.
- `PatchDataset`: deterministic overlapping patch view over datasets that return `(image, mask, metadata)`.

Every segmentation-style dataset returns:

| Position | Object | Contract |
| --- | --- | --- |
| `0` | `image` | `torch.float32`, `[C,H,W]`, finite after loader preprocessing |
| `1` | `mask` | `torch.uint8`, `[1,H,W]`, binary, or `None` in `mode="none"` |
| `2` | `metadata` | `dict` with provenance and `valid_mask` unless intentionally absent |

Strict multimodal pretraining uses a different sample shape from `FusedDataset(return_mode="paired")`:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `sar` | `[C_sar,H,W]` | `float32` | SAR tensor only |
| `optical` | `[C_opt,H,W]` | `float32` | optical tensor only |
| `valid_mask` | `[H,W]` | `bool` | intersection of SAR and optical valid pixels |
| `meta` | `dict` | - | non-label-bearing paired provenance |

## Role in the Overall Pipeline

The data loader is the boundary between geospatial assets and model code. It is responsible for:

- resolving IDs or paths into readable raster files;
- reading image bands in channel-first order;
- converting image values into finite floating-point tensors;
- loading masks only when the requested mode is labeled;
- preserving pixel-validity information for loss masking;
- preserving sample provenance for debugging;
- enforcing strict SAR/optical pairing when requested;
- batching samples without losing metadata alignment.

Model and training code should not reopen rasters, infer file pairing, or reinterpret label ownership. Those decisions belong here.

Known training interactions in this workspace:

- `src/Multi_modal_src/train.py` uses `FusedDataset.from_combined_manifest(..., return_mode="paired")` and `multimodal_pretrain_collate`.
- `src/Multi_modal_src/eval.py` expects that collated paired batches contain `sar`, `optical`, `valid_mask`, and `meta`, then builds two augmented views per modality.
- `src/train.py` and `src/eval.py` were requested for tracing but do not exist at those exact paths in this repository checkout. This absence is documented rather than inferred.

## File-by-File Map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Public package export surface for datasets, collate functions, transforms, splits, stats, discovery, and validation. |
| `augmentations.py` | Stateless tensor transforms that preserve spatial alignment across image, mask, `valid_mask`, and `ignore_mask`. |
| `collate.py` | Batch assembly for segmentation samples and paired multimodal samples. |
| `combined_manifest.py` | Strict `Combined/manifest.csv` parsing, asset resolution, and paired asset validation. |
| `discover_ids.py` | Directory stem discovery and image/mask ID intersection utilities. |
| `example_optical_fused.py` | Runnable examples for optical-only, strict fused, and legacy fused workflows. |
| `example_usage.py` | SAR-focused quickstart showing splits, stats, DataLoader, and loss masking. |
| `filter_south_asia.py` | Utility script that copies country-prefixed files into a filtered dataset tree. |
| `fused_dataset.py` | SAR/optical pairing, fused channel concatenation, paired pretraining samples, and strict manifest construction. |
| `optical_dataset.py` | Sentinel-2 reading, scaling, clipping, cloud invalidation, normalization, mask loading. |
| `patch_dataset.py` | Deterministic full-coverage overlapping patch index and metadata propagation. |
| `sar_dataset.py` | Sentinel-1 reading, clipping, optional log transform, normalization, mask loading, optional time-matched stack metadata. |
| `splits.py` | Deterministic and stratified train/validation ID splitting. |
| `stats.py` | Streaming per-band mean/std over valid pixels only. |
| `validate_dataset.py` | Runtime shape, dtype, finiteness, channel, and mask sanity checks. |

Detailed per-file documentation lives in `src/data_loader/docs/`.

## Full Data Flow

1. Raw assets are stored as `.tif` or `.tiff` rasters.
2. IDs are discovered from directories, passed directly by callers, or loaded from `Combined/manifest.csv`.
3. A dataset resolves each ID to an image path and, in labeled modes, a mask path.
4. Rasterio reads image bands as `[C,H,W]`.
5. Modality-specific preprocessing converts raw values to finite `float32` tensors.
6. A pixel-level `valid_mask` is created from non-finite, nodata, cloud, or transform-invalidated pixels.
7. Optional masks are loaded as binary `[1,H,W]` `uint8` tensors.
8. Optional dataset transforms mutate image, mask, `valid_mask`, and `ignore_mask` together.
9. `PatchDataset`, if used, crops image, mask, `valid_mask`, and `ignore_mask` together and extends metadata.
10. A collate function stacks tensors and preserves metadata lists.
11. Training code consumes tensors and should apply `valid_mask` to pixel losses or pretraining views.

Segmentation path:

```text
raw S1/S2 raster + optional label
  -> SARDataset or OpticalDataset
  -> optional transforms
  -> optional PatchDataset
  -> default_collate
  -> model + loss masked by valid_masks
```

Strict multimodal pretraining path:

```text
Combined/manifest.csv
  -> load_combined_manifest_samples
  -> SARDataset(paths) + OpticalDataset(paths)
  -> FusedDataset(strict_pairing=True, return_mode="paired")
  -> multimodal_pretrain_collate
  -> Multi_modal_src.train.build_decur_batch_views
  -> DeCUR model
```

## Dataset Contract

### Segmentation Tuple Contract

Returned Sample Schema:

| Key/Position | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `image` / `0` | `[C,H,W]` | `float32` | Finite channel-first model input after modality preprocessing. |
| `mask` / `1` | `[1,H,W]` or `None` | `uint8` | Binary flood label for labeled modes; `None` for unlabeled mode. |
| `metadata` / `2` | `dict` | - | Provenance, modality, path fields, and pixel validity. |
| `metadata["valid_mask"]` | `[H,W]` or occasionally `[1,H,W]` after wrappers | `bool` | `True` means pixel is valid for loss/view use. |
| `metadata["ignore_mask"]` | `[1,H,W]` or `[H,W]`, optional | `bool` | Pixels whose labels should be ignored, usually weak-label boundaries. |

Invariants:

- `image.shape[-2:] == mask.shape[-2:]` when `mask is not None`.
- `metadata["valid_mask"].shape[-2:] == image.shape[-2:]` when present.
- `image` should be finite after preprocessing.
- `mask` is label data. `valid_mask` is data-quality and augmentation validity data. They are not interchangeable.

### Paired Multimodal Contract

`FusedDataset(return_mode="paired")` returns a dictionary rather than a segmentation tuple:

Returned Sample Schema:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `sar` | `[C_sar,H,W]` | `float32` | Sentinel-1 input. |
| `optical` | `[C_opt,H,W]` | `float32` | Sentinel-2 input. |
| `valid_mask` | `[H,W]` | `bool` | `sar_valid & optical_valid`. |
| `meta` | `dict` | - | Paired provenance. Must not include `mask_path` or `label_path` for pretraining collate. |

`multimodal_pretrain_collate` converts `valid_mask` to `[B,1,H,W]`.

## Valid Mask Philosophy

`valid_mask` answers: "Can this pixel safely contribute to the objective?"

It is not a flood label, nodata class, or background mask. `True` means the input values at that pixel are trustworthy after preprocessing and augmentation. `False` means a pixel should be excluded from pixel losses or zeroed before contrastive view construction.

Current invalidation sources:

- SAR non-finite values after read.
- SAR non-positive values before log transform.
- Optical non-finite values after read.
- Optical all-zero pixels across all loaded bands.
- Optional optical cloud/SCL classes.
- Spatial transforms that move pixels outside the source extent.
- `RandomCutout`, which marks artificial blank regions invalid.

ENFORCED:

- `FusedDataset` coerces SAR and optical valid masks to `bool` and checks spatial shape.
- `default_collate` extracts metadata `valid_mask` and stacks it separately.
- `multimodal_pretrain_collate` requires a `valid_mask` key and normalizes it to `[1,H,W]`.

ASSUMED:

- Training loops will actually apply the mask. The loader provides it; it cannot enforce loss usage.
- `valid_mask=True` means all required modality channels for the current sample are valid.

ACCIDENTAL:

- Some helper code accepts both `[H,W]` and `[1,H,W]`; this is practical compatibility, not a clean single canonical representation.

## Mask / Label Ownership Rules

- `SARDataset` and `OpticalDataset` own label loading in labeled modes.
- `FusedDataset` uses the SAR-side mask as the fused segmentation label.
- Strict multimodal pretraining uses `mode="none"` and removes label-bearing metadata from collated batches.
- `PatchDataset` crops an already-loaded mask but does not reinterpret labels.
- Augmentations may spatially transform masks or erode weak labels, but radiometric transforms must not alter masks.

ENFORCED:

- Labeled dataset modes require `mask_root`.
- Mask shapes are checked against image shape when loaded.
- `multimodal_pretrain_collate` rejects `mask_path` and `label_path` inside paired `meta`.

ASSUMED:

- A positive mask pixel means flood/water label, and all positive raster values can be binarized as `1`.
- SAR and optical masks, if both exist outside strict mode, would represent the same label semantics. The fused path deliberately keeps SAR mask ownership to avoid ambiguity.

## SAR vs Optical vs Fused Responsibilities

SAR loader responsibilities:

- read Sentinel-1 bands;
- optionally robust-clip each band using 2nd/98th percentiles;
- optionally apply log transform;
- flag non-finite and log-invalid pixels;
- impute non-finite values for model safety;
- optionally attach time-matched 8-band stacks in metadata.

Optical loader responsibilities:

- read all or selected Sentinel-2 bands;
- detect DN scale versus reflectance scale;
- scale/clamp to `[0,1]`;
- robust-clip reflectance bands;
- flag all-zero nodata pixels;
- optionally extend invalidity with cloud/SCL masks;
- impute non-finite values for model safety.

Fused loader responsibilities:

- pair SAR and optical samples by ID or strict manifest row;
- enforce strict order and asset parity in strict mode;
- concatenate channels for segmentation-style fused samples;
- keep separate modality tensors for paired pretraining;
- compose valid masks with logical AND when both modalities exist;
- preserve provenance for debugging.

## Strict Pairing and Manifest Logic

Strict pairing is created only through `FusedDataset.from_combined_manifest`.

ENFORCED:

- `manifest.csv` must exist under the combined root unless the caller passes the manifest path directly.
- Manifest rows must include `sample_id`, `output_S1`, and `output_S2`; `output_Label` is required when `mode != "none"`.
- Asset paths must live under the expected `S1/`, `S2/`, or `Label/` subdirectory.
- Asset file stems must equal `sample_id`.
- Referenced assets must exist.
- Optional validation checks raster readability and SAR/optical spatial shape equality.
- Strict `FusedDataset` asserts SAR IDs, optical IDs, manifest IDs, and child dataset path order all match exactly.
- Strict mode requires `optical_missing_policy="raise"`.

ASSUMED:

- Manifest row order is the canonical sample order.
- The combined dataset has already been geospatially aligned; validation checks raster shapes, not geotransform equality.
- Matching file stems imply matching semantic events/times unless external manifest creation guaranteed more.

ACCIDENTAL:

- Non-strict `FusedDataset` builds an optical ID lookup from `optical_dataset.samples` when available. Duplicate IDs overwrite earlier entries.

## PatchDataset vs Full-Tile Behavior

Full-tile datasets return one sample per raster. `PatchDataset` wraps such a dataset and exposes square crops as independent samples.

Patch behavior:

- `stride = ceil(patch_size * (1 - overlap))`, minimum `1`;
- edge starts are clamped so the final row and column cover the image boundary;
- image, mask, `valid_mask`, and `ignore_mask` are cropped with identical coordinates;
- metadata gets `base_id`, `patch_row`, `patch_col`, `patch_y0`, `patch_x0`, and `patch_size`;
- `id` becomes `<base_id>_r<row>_c<col>`.

ASSUMED:

- The base dataset returns full-resolution tensors at least as large as requested patches or slicing will produce smaller-than-expected patches for very small rasters.

ACCIDENTAL:

- Constructor arguments `skip_mostly_nodata` and `nodata_threshold` are stored but not currently used to filter patches in `_build_index`.

## Augmentation Boundaries and Guarantees

Transforms operate on tensors, not files. Their callable signature is:

```python
transform(image: torch.Tensor, mask: Optional[torch.Tensor], metadata: dict)
    -> tuple[torch.Tensor, Optional[torch.Tensor], dict]
```

ENFORCED:

- Spatial transforms apply matching geometry to image, mask, `valid_mask`, and `ignore_mask`.
- Image interpolation is bilinear for affine-like transforms; mask interpolation is nearest.
- `WeakLabelBoundaryErosion` only mutates masks and `ignore_mask`.
- `RandomCutout` updates `valid_mask` so artificial blank regions can be excluded.

ASSUMED:

- Modality selection passed to `build_train_transforms(modality=...)` matches the actual tensor. Passing `modality="fused"` applies some transforms across all channels, including SAR and optical channels together.
- Random transforms are acceptable for `DataLoader` workers because transforms are stateless except constructor parameters.

ACCIDENTAL:

- `RandomChannelDrop` computes `random.randint(1, min(max_drop, n_channels - 1))`; one-channel tensors can fail because the upper bound becomes `0`.
- `RandomCropAndResize` and `_apply_spatial_transform` assume `valid_mask` is 2-D. Some other code accepts `[1,H,W]`.

## Interaction With Training Code

`src/Multi_modal_src/train.py`:

- imports `FusedDataset`, `build_train_transforms`, `build_val_transforms`, and `multimodal_pretrain_collate`;
- requires exactly 2 SAR channels and 13 optical channels by default;
- constructs `FusedDataset.from_combined_manifest(..., mode="none", return_mode="paired")`;
- optionally validates the manifest with `--validate-manifest`;
- splits strict paired samples after dataset construction via `random_split`;
- uses `multimodal_pretrain_collate` in `DataLoader`;
- applies modality-specific train transforms in `build_decur_batch_views`, not inside the child datasets.

`src/Multi_modal_src/eval.py`:

- requires collated paired batches with `sar`, `optical`, `valid_mask`, and `meta`;
- creates two SAR views and two optical views;
- propagates transformed valid masks through the augmentation metadata;
- masks invalid pixels by multiplying views by valid masks before the model forward pass.

No `src/train.py` or `src/eval.py` file exists at the requested paths in this checkout.

## Extension Guidelines

### How to Add a New Modality

1. Implement a dataset that returns `(image, mask, metadata)` with `[C,H,W]` finite `float32` images.
2. Define invalid pixel semantics and put a `bool` `[H,W]` `valid_mask` in metadata.
3. Keep label ownership explicit. Do not silently borrow labels from another modality unless a wrapper documents that rule.
4. Add a collate path only if the sample schema is not compatible with `default_collate`.
5. Add validation checks for channel count, dtype, finite values, and mask shape.
6. Decide whether strict pairing requires a manifest or whether ID lookup is acceptable.

### How to Safely Modify `__getitem__`

Before changing any `__getitem__`:

- preserve return schema or update all collate/training consumers;
- preserve `valid_mask` shape and meaning;
- preserve path/provenance metadata;
- keep masks binary `[1,H,W]` `uint8` when present;
- check `PatchDataset`, `default_collate`, `validate_sample_shapes`, and `Multi_modal_src/eval.py` for hidden assumptions;
- run a small DataLoader batch through the intended collate function.

## Known Fragile Areas

- Shape convention drift between `[H,W]` and `[1,H,W]` valid masks.
- Non-strict fused ID lookup can silently overwrite duplicate optical IDs.
- Strict validation checks spatial dimensions but not CRS, transform, pixel grid, or acquisition date.
- `PatchDataset.skip_mostly_nodata` is currently not implemented despite being exposed.
- `RandomChannelDrop` is unsafe for one-channel tensors.
- `compute_running_mean_std` calls `vm.numpy()` without detaching or moving to CPU, assuming metadata masks are CPU tensors.
- `example_usage.py` imports from `dataloader.*`, while this package path is `data_loader.*` elsewhere.
- `stats.py` and comments contain mojibake characters from encoding drift; this does not affect runtime but hurts readability.

## Common Debugging Checklist

- Can `rasterio.open(path)` read every image and mask?
- Does `len(dataset)` match expected manifest or ID count?
- Does one sample have finite `image` values?
- Is `mask is None` only in `mode="none"`?
- Does `metadata["valid_mask"].shape[-2:] == image.shape[-2:]`?
- Does `default_collate` return `valid_masks` with `[B,H,W]`?
- Does `multimodal_pretrain_collate` return `valid_mask` with `[B,1,H,W]`?
- Are SAR and optical channel counts what the model expects?
- For strict fused mode, does `metadata["strict_paired_mode"]` equal `True`?
- For patching, do patched metadata coordinates map back to the source tile?

## Failure Case Examples

- A `Combined/manifest.csv` row has `output_S2=S2/other_id.tif`: strict manifest loading raises because the stem does not equal `sample_id`.
- A SAR image is 512x512 and its mask is 513x512: `_load_mask` raises a shape mismatch.
- A batch mixes labeled and unlabeled samples: `default_collate` raises instead of returning a partial mask batch.
- A paired pretraining sample includes `label_path` in `meta`: `multimodal_pretrain_collate` raises because pretraining batches must not carry label-bearing metadata.
- Optical SCL files are missing: `OpticalDataset` warns and skips cloud masking for that tile, but keeps the sample.

## Recommended Reading Order

1. `combined_manifest.py`
2. `sar_dataset.py`
3. `optical_dataset.py`
4. `fused_dataset.py`
5. `collate.py`
6. `augmentations.py`
7. `patch_dataset.py`
8. `stats.py` and `validate_dataset.py`
9. Examples and utility scripts

## Checklist Before Modifying Loader

- Identify whether the change affects tuple samples, paired dict samples, or both.
- State whether the change is label-related, validity-related, modality-related, or metadata-only.
- Verify all shape and dtype contracts after the change.
- Keep strict manifest behavior strict; do not add silent fallback to strict mode.
- Preserve valid-mask composition rules for fused data.
- Preserve augmentation alignment across image, mask, `valid_mask`, and `ignore_mask`.
- Update the matching per-file doc under `src/data_loader/docs/`.
- Run at least a one-sample dataset access and one DataLoader batch through the relevant collate function.
