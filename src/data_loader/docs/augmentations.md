# `augmentations.py`

## Purpose

Defines stateless tensor augmentation transforms for SAR, optical, fused segmentation, and multimodal pretraining views. The key responsibility is preserving spatial alignment between `image`, `mask`, `metadata["valid_mask"]`, and `metadata["ignore_mask"]`.

## Why This File Exists

Flood segmentation and multimodal contrastive training need augmentation, but geospatial masks and validity masks cannot be treated like ordinary images. This module centralizes transform behavior so masks use nearest-neighbor geometry, images use appropriate interpolation, and invalid/ignored pixels remain aligned with model inputs.

## Dependencies

Internal: none.

External:

- `math`, `random`
- `torch`
- `torch.nn.functional as F`

## Classes

### `_Transform`

Responsibility: base class storing probability `p`, probability sampling, abstract `__call__`, and a readable `__repr__`.

Constructor arguments:

- `p`: probability in `[0,1]`.

Internal state: `self.p`.

Public API:

- `_should_apply()`
- `__call__(img, mask, meta)` abstract
- `__repr__()`

### `RandomHorizontalFlip`

Responsibility: flip image, optional mask, `valid_mask`, and `ignore_mask` horizontally.

`__getitem__` behavior: not a dataset.

Transform behavior:

- Image flips along width axis.
- Mask flips along width axis.
- `valid_mask` and `ignore_mask` axes are mapped to their spatial dimensions.

### `RandomVerticalFlip`

Responsibility: flip image, optional mask, `valid_mask`, and `ignore_mask` vertically.

### `RandomRotation90`

Responsibility: rotate by random `k` in `{1,2,3}` using `torch.rot90`.

Design note: 90-degree rotations preserve grid values exactly and avoid interpolation artifacts.

### `RandomRotationFine`

Responsibility: apply small continuous rotation in `[-max_angle,+max_angle]`.

Constructor arguments:

- `max_angle`: max degrees.
- `p`: probability.

Behavior:

- Image uses bilinear sampling and border padding.
- Masks use nearest sampling and zero/false padding.
- Output size equals input size.

### `RandomCropAndResize`

Responsibility: crop a random area/aspect ratio and resize back to original size.

Constructor arguments:

- `scale`: area fraction range.
- `ratio`: aspect ratio range.
- `p`: probability.

Behavior:

- Attempts 10 valid crops.
- Image is bilinear-resized.
- Mask, `valid_mask`, and `ignore_mask` use nearest-resize.
- Falls back to unchanged sample if no valid crop is found.

### `RandomSpeckleNoise`

Responsibility: add SAR-aware noise approximating multiplicative speckle in log space.

Constructor arguments:

- `sigma`: positive noise std.
- `p`: probability.

Behavior: returns `img + torch.randn_like(img) * sigma`; masks and metadata unchanged.

### `RandomAdditiveGaussianNoise`

Responsibility: add sampled Gaussian noise to image.

Constructor arguments:

- `std_range`: uniform range for noise std.
- `p`: probability.

### `RandomBrightnessContrast`

Responsibility: apply per-sample additive brightness and multiplicative contrast in feature space.

Constructor arguments:

- `brightness`
- `contrast`
- `p`

### `RandomChannelDrop`

Responsibility: zero out random channels.

Constructor arguments:

- `max_drop`: maximum number of channels to drop, must be at least 1.
- `p`: probability.

Risk: one-channel tensors can fail because `random.randint(1, min(max_drop, n_channels - 1))` has an invalid upper bound when `n_channels == 1`.

### `RandomGamma`

Responsibility: apply a random gamma/power transform while avoiding fractional powers of negative values.

Constructor arguments:

- `gamma_range`
- `p`

Behavior:

- Shifts image by its minimum to make values non-negative.
- Applies gamma.
- Shifts output mean approximately back to original mean.

### `RandomCutout`

Responsibility: zero a random rectangular image region and mark that region invalid in `valid_mask`.

Constructor arguments:

- `max_frac`
- `fill_value`
- `p`

Design rationale: cutout is artificial missing data, so it should not contribute to pixel loss.

### `WeakLabelBoundaryErosion`

Responsibility: erode weak-label flood masks and mark eroded boundary pixels in `ignore_mask`.

Constructor arguments:

- `erosion_px`
- `create_ignore`
- `p`

Behavior:

- No-op when mask is `None` or probability does not apply.
- Uses max-pooling to compute dilated and eroded masks.
- Replaces mask with eroded mask.
- Creates or extends `metadata["ignore_mask"]` with uncertain boundary pixels when `create_ignore=True`.

### `Compose`

Responsibility: apply an ordered list of transforms.

Constructor arguments:

- `transforms`: list of callable transforms.

Public API:

- `__call__(img, mask, meta)`
- `__repr__`

## Functions

### `_to_float_chw(mask)`

Purpose: convert `[1,H,W]` mask to `[1,1,H,W]` float tensor for grid sampling.

### `_from_float_chw(t, orig_dtype)`

Purpose: convert `[1,1,H,W]` sampled tensor back to `[1,H,W]` original dtype.

### `_apply_spatial_transform(img, mask, meta, theta, out_h, out_w)`

Purpose: apply a 2x3 affine transform to image, mask, `valid_mask`, and `ignore_mask`.

Inputs:

- `img`: `[C,H,W]`
- `mask`: optional `[1,H,W]`
- `meta`: dict
- `theta`: `[1,2,3]`
- output dimensions

Outputs: transformed `(img, mask, meta)`.

Side effects: shallow-copies metadata before mutation.

Preconditions:

- `valid_mask`, if present, is expected as `[H,W]`.
- `ignore_mask`, if present, may be `[H,W]` or `[1,H,W]`.

### `_flip_tensor(t, dim)`

Purpose: flip optional tensor or return `None`.

### `_flip_meta_masks(meta, img_dim)`

Purpose: flip `valid_mask` and `ignore_mask` consistently with image flip.

### `build_train_transforms(...)`

Purpose: return default train `Compose` for `modality in {"sar","optical","fused"}`.

Inputs:

- modality, weak-label erosion controls, noise/jitter/cutout/channel-drop parameters.

Outputs: `Compose`.

Pipeline:

1. Optional `WeakLabelBoundaryErosion`.
2. Horizontal flip.
3. Vertical flip.
4. 90-degree rotation.
5. Fine rotation.
6. Crop and resize.
7. Modality-gated speckle noise for SAR/fused.
8. Modality-gated gamma for optical/fused.
9. Brightness/contrast.
10. Additive Gaussian noise.
11. Modality-gated channel drop for SAR/fused.
12. Cutout.

Where it is used:

- Imported by `src/Multi_modal_src/train.py`, where separate SAR and optical view transforms are created.

### `build_val_transforms()`

Purpose: return identity `Compose([])` for validation/inference symmetry.

## Data Contracts

Transform callable contract:

```python
(img: torch.Tensor, mask: Optional[torch.Tensor], meta: dict)
    -> tuple[torch.Tensor, Optional[torch.Tensor], dict]
```

Returned Sample Schema after transforms:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `image` | `[C,H,W]` unless transform changes size by design | float tensor | Augmented image. |
| `mask` | `[1,H,W]` or `None` | original dtype, typically `uint8` | Spatially aligned label. |
| `valid_mask` | `[H,W]` usually | bool | Spatially aligned validity mask. |
| `ignore_mask` | `[1,H,W]` or `[H,W]` | bool | Spatially aligned ignored label region. |

## Tensor Shape Expectations

- Images are channel-first `[C,H,W]`.
- Masks are `[1,H,W]`.
- Valid masks are mostly assumed `[H,W]`.
- Ignore masks may be `[H,W]` or `[1,H,W]`.

## Metadata Structure

Transforms shallow-copy metadata only when they mutate it. Metadata keys used:

- `valid_mask`
- `ignore_mask`
- `modality` is documented conceptually but most class-level transforms do not inspect it; gating happens in `build_train_transforms`.

## Valid Mask Handling

Spatial transforms move `valid_mask` with nearest-neighbor logic. `RandomCutout` marks artificial blank regions invalid. Radiometric transforms do not change `valid_mask`.

ENFORCED:

- Geometric transforms keep masks aligned by applying the same spatial operation.
- Fine rotations use nearest-neighbor masks and false padding for validity/ignore regions.

ASSUMED:

- `valid_mask` is 2-D for several transforms (`_apply_spatial_transform`, `RandomCropAndResize`, `RandomCutout`).
- Any invalid pixel should remain excluded after geometric movement.

ACCIDENTAL:

- Shape convention support is inconsistent: flips handle `[H,W]` and `ignore_mask` variants well, but some transforms assume 2-D `valid_mask`.

## Transform / Augmentation Behavior

Geometry affects image, mask, validity, and ignore masks. Radiometric/noise transforms affect only image. Weak-label erosion affects mask and ignore mask. Cutout affects image and valid mask.

## Design Decisions

- Transforms are stateless/picklable so DataLoader workers can use them.
- SAR/fused get speckle and channel drop; optical/fused get gamma.
- The fused modality applies some transforms across all channels at once, which maintains channel alignment but may be physically imperfect for mixed SAR/optical channels.

## Edge Cases and Failure Modes

- `RandomChannelDrop` can fail for one-channel images.
- `RandomCutout` can fail if computed `cut_h` or `cut_w` is zero or equals dimensions in unexpected ways for very small images.
- `WeakLabelBoundaryErosion` assumes binary-ish masks.
- Continuous rotation pads masks with zero/false, which may remove labels near edges.

## Modification Risks

- Any spatial transform that forgets `valid_mask` or `ignore_mask` silently corrupts training.
- Using bilinear interpolation for masks would create non-binary labels.
- Applying optical color transforms to SAR channels can produce physically meaningless inputs.
- Changing transform order can alter weak-label ignore propagation.

## Example Usage

```python
sar_transform = build_train_transforms(modality="sar")
opt_transform = build_train_transforms(modality="optical")
img2, mask2, meta2 = sar_transform(img, mask, meta)
```

## Open Questions / Ambiguities

- Whether fused data should have modality-aware channel groups for radiometric transforms.
- Whether all transforms should normalize `valid_mask` to one canonical shape first.
- Whether channel drop should be disabled or fixed for one-channel inputs.

