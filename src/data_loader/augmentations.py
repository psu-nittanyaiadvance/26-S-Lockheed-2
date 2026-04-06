"""
augmentations.py  –  Image augmentation pipeline for SAR (and fused) flood segmentation.

Design principles
-----------------
1. **Geometric consistency**: every spatial transform (flip, rotate, crop) is
   applied identically to the image tensor, the mask, valid_mask, and
   ignore_mask.  Masks use nearest-neighbour interpolation; images use
   bilinear.

2. **SAR-aware radiometric augmentations**: SAR backscatter is physics-based.
   We support:
   - Multiplicative speckle noise  (dominant noise model for SAR GRD data)
   - Additive Gaussian noise       (thermal noise floor)
   - Per-band brightness / contrast jitter
   These are NOT applied to the segmentation mask.

3. **Optical radiometric augmentations**: colour jitter (brightness, contrast,
   saturation) is gated on ``modality="optical"`` so it is never applied to
   SAR or fused data where SAR bands would be nonsensically altered.

4. **Label-noise robustness for weak labels**: a ``weak_label_erode`` flag
   can erode the boundary of the weak (Otsu) masks slightly, matching the
   unreliable-boundary characteristic of threshold-based labelling.

5. **valid_mask propagation**: all spatial transforms slice valid_mask in sync
   so invalid pixels (NaN/Inf in the original tile) remain excluded from loss.

6. **Stateless / picklable**: every transform is a pure callable with no
   mutable state beyond constructor hyper-parameters.  DataLoader workers can
   pickle the pipeline.

7. **Composable**: ``Compose`` wraps a list of individual transforms, each
   with a probability ``p``.  The top-level ``build_train_transforms`` and
   ``build_val_transforms`` factory functions return sensible defaults.

Usage
-----
    from src.data_loader.augmentations import build_train_transforms, build_val_transforms

    train_transforms = build_train_transforms(modality="sar")
    val_transforms   = build_val_transforms()

    train_ds = SARDataset(
        ...,
        transforms=train_transforms,
    )

Transform callable signature
-----------------------------
    __call__(
        img:    torch.Tensor  [C, H, W]  float32,
        mask:   Optional[torch.Tensor]   [1, H, W]  uint8,
        meta:   dict                     (valid_mask [H,W] bool, optional ignore_mask)
    ) -> (img, mask, meta)

Every transform returns the same triple so they can be stacked in Compose.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_float_chw(mask: torch.Tensor) -> torch.Tensor:
    """[1, H, W] uint8  →  [1, 1, H, W] float32 (for grid_sample / affine_grid)."""
    return mask.unsqueeze(0).float()


def _from_float_chw(t: torch.Tensor, orig_dtype: torch.dtype) -> torch.Tensor:
    """[1, 1, H, W] float  →  [1, H, W] orig_dtype."""
    return t.squeeze(0).to(orig_dtype)


def _apply_spatial_transform(
    img: torch.Tensor,
    mask: Optional[torch.Tensor],
    meta: Dict[str, Any],
    theta: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    Apply a 2×3 affine matrix to img, mask, valid_mask, and ignore_mask.

    Parameters
    ----------
    theta : torch.Tensor   shape [1, 2, 3]
    out_h, out_w : int     output spatial size
    """
    # ── image (bilinear) ──────────────────────────────────────────────────────
    img4 = img.unsqueeze(0)  # [1, C, H, W]
    grid = F.affine_grid(theta, (1, img.shape[0], out_h, out_w), align_corners=False)
    img_t = F.grid_sample(img4, grid, mode="bilinear", padding_mode="border",
                          align_corners=False).squeeze(0)

    # ── mask (nearest) ────────────────────────────────────────────────────────
    mask_t: Optional[torch.Tensor] = None
    if mask is not None:
        mask4 = _to_float_chw(mask)
        grid_m = F.affine_grid(theta, (1, 1, out_h, out_w), align_corners=False)
        mask_t = F.grid_sample(mask4, grid_m, mode="nearest", padding_mode="zeros",
                               align_corners=False)
        mask_t = _from_float_chw(mask_t, mask.dtype)

    meta = dict(meta)  # shallow copy so we don't mutate caller's dict

    # ── valid_mask (nearest) ──────────────────────────────────────────────────
    vm = meta.get("valid_mask")
    if vm is not None:
        if not isinstance(vm, torch.Tensor):
            vm = torch.as_tensor(vm)
        vm4 = vm.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        grid_v = F.affine_grid(theta, (1, 1, out_h, out_w), align_corners=False)
        vm_t = F.grid_sample(vm4, grid_v, mode="nearest", padding_mode="zeros",
                             align_corners=False)
        meta["valid_mask"] = vm_t.squeeze(0).squeeze(0).bool()

    # ── ignore_mask (nearest) ─────────────────────────────────────────────────
    ig = meta.get("ignore_mask")
    if ig is not None:
        if not isinstance(ig, torch.Tensor):
            ig = torch.as_tensor(ig)
        # normalise to [H, W]
        if ig.ndim == 3:
            ig = ig.squeeze(0)
        ig4 = ig.unsqueeze(0).unsqueeze(0).float()
        grid_i = F.affine_grid(theta, (1, 1, out_h, out_w), align_corners=False)
        ig_t = F.grid_sample(ig4, grid_i, mode="nearest", padding_mode="zeros",
                             align_corners=False)
        # restore to [1, H, W] as expected downstream
        meta["ignore_mask"] = ig_t.squeeze(0).bool()

    return img_t, mask_t, meta


def _flip_tensor(
    t: Optional[torch.Tensor],
    dim: int,
) -> Optional[torch.Tensor]:
    return None if t is None else t.flip(dim)


def _flip_meta_masks(meta: Dict[str, Any], img_dim: int) -> Dict[str, Any]:
    """
    Flip spatial masks consistently with an image flip along ``img_dim``.

    ``img_dim`` is the dimension index on the image tensor [C, H, W]:
      - img_dim = -2  →  flip rows  (vertical flip)
      - img_dim = -1  →  flip cols  (horizontal flip)

    ``valid_mask`` is [H, W] (2-D): row axis = 0 / -2, col axis = 1 / -1.
    ``ignore_mask`` is [1, H, W] (3-D): row axis = 1 / -2, col axis = 2 / -1.
    """
    meta = dict(meta)

    vm = meta.get("valid_mask")
    if vm is not None:
        # vm is 2-D [H, W]; img_dim -2 → row axis 0; img_dim -1 → col axis 1
        vm_dim = 0 if img_dim == -2 else 1
        meta["valid_mask"] = vm.flip(vm_dim)

    ig = meta.get("ignore_mask")
    if ig is not None:
        if not isinstance(ig, torch.Tensor):
            ig = torch.as_tensor(ig)
        # ig is [1, H, W] or [H, W]; flip along the spatial axis that matches img_dim
        if ig.ndim == 3:
            ig_dim = 1 if img_dim == -2 else 2   # [1, H, W]: H=1, W=2
        else:
            ig_dim = 0 if img_dim == -2 else 1   # [H, W]: H=0, W=1
        meta["ignore_mask"] = ig.flip(ig_dim)

    return meta


# ── base class ────────────────────────────────────────────────────────────────

class _Transform:
    """Base class: stores ``p`` and provides ``__repr__``."""

    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}")
        self.p = p

    def _should_apply(self) -> bool:
        return random.random() < self.p

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in vars(self).items()
            if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"


# ── geometric transforms ──────────────────────────────────────────────────────

class RandomHorizontalFlip(_Transform):
    """Flip image (and all spatial masks) horizontally with probability p."""

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta
        # dim=-1 = W axis for [C, H, W]
        img = img.flip(-1)
        mask = _flip_tensor(mask, -1)
        meta = _flip_meta_masks(meta, img_dim=-1)
        return img, mask, meta


class RandomVerticalFlip(_Transform):
    """Flip image (and all spatial masks) vertically with probability p."""

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta
        # dim=-2 = H axis for [C, H, W]
        img = img.flip(-2)
        mask = _flip_tensor(mask, -2)
        meta = _flip_meta_masks(meta, img_dim=-2)
        return img, mask, meta


class RandomRotation90(_Transform):
    """
    Rotate by a random multiple of 90° (k ∈ {1, 2, 3}) with probability p.

    SAR backscatter is not rotationally invariant (the sensor look-direction
    is fixed), but 90°-step rotations are a widely-used augmentation in
    remote-sensing segmentation and expand the apparent variety of flood
    geometry without distorting backscatter statistics.
    """

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta
        k = random.randint(1, 3)
        img = torch.rot90(img, k, dims=(-2, -1))
        if mask is not None:
            mask = torch.rot90(mask, k, dims=(-2, -1))
        meta = dict(meta)
        vm = meta.get("valid_mask")
        if vm is not None:
            meta["valid_mask"] = torch.rot90(vm, k, dims=(-2, -1))
        ig = meta.get("ignore_mask")
        if ig is not None:
            meta["ignore_mask"] = torch.rot90(ig, k, dims=(-2, -1))
        return img, mask, meta


class RandomRotationFine(_Transform):
    """
    Rotate by a small continuous angle in ``[-max_angle, +max_angle]`` degrees.

    Bilinear interpolation for the image; nearest for all masks.  Pixels that
    fall outside the original extent after rotation are filled with the border
    value (images) or zero / False (masks).

    Parameters
    ----------
    max_angle : float
        Maximum rotation angle in degrees.  Default: 15°.
    p : float
        Probability of applying the transform.
    """

    def __init__(self, max_angle: float = 15.0, p: float = 0.3) -> None:
        super().__init__(p)
        self.max_angle = max_angle

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta

        angle_deg = random.uniform(-self.max_angle, self.max_angle)
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Rotation matrix (no translation): [[cos, -sin, 0], [sin, cos, 0]]
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 2, 3]

        h, w = img.shape[-2], img.shape[-1]
        return _apply_spatial_transform(img, mask, meta, theta, out_h=h, out_w=w)


class RandomCropAndResize(_Transform):
    """
    Randomly crop a ``scale`` fraction of the image area, then resize back to
    the original resolution.

    This simulates scale variation for flood patches that differ greatly in
    size (urban puddles vs large river inundations).

    Parameters
    ----------
    scale : tuple of float
        Min/max fraction of the original area to crop.  Default: (0.7, 1.0).
    ratio : tuple of float
        Min/max aspect ratio of the crop.  Default: (0.9, 1.1).
    p : float
        Probability of applying the transform.
    """

    def __init__(
        self,
        scale: Tuple[float, float] = (0.7, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.scale = scale
        self.ratio = ratio

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta

        _, H, W = img.shape
        area = H * W
        for _ in range(10):  # retry loop
            target_area = random.uniform(*self.scale) * area
            log_ratio = random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect = math.exp(log_ratio)
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= W and 0 < crop_h <= H:
                x0 = random.randint(0, W - crop_w)
                y0 = random.randint(0, H - crop_h)
                # Crop then resize
                img_crop = img[:, y0:y0 + crop_h, x0:x0 + crop_w]
                img_out = F.interpolate(
                    img_crop.unsqueeze(0), size=(H, W), mode="bilinear",
                    align_corners=False
                ).squeeze(0)

                meta = dict(meta)
                mask_out: Optional[torch.Tensor] = None
                if mask is not None:
                    m_crop = mask[:, y0:y0 + crop_h, x0:x0 + crop_w]
                    mask_out = F.interpolate(
                        m_crop.unsqueeze(0).float(), size=(H, W), mode="nearest"
                    ).squeeze(0).to(mask.dtype)

                vm = meta.get("valid_mask")
                if vm is not None:
                    vm_crop = vm[y0:y0 + crop_h, x0:x0 + crop_w]
                    meta["valid_mask"] = F.interpolate(
                        vm_crop.unsqueeze(0).unsqueeze(0).float(), size=(H, W),
                        mode="nearest"
                    ).squeeze(0).squeeze(0).bool()

                ig = meta.get("ignore_mask")
                if ig is not None:
                    if ig.ndim == 3:
                        ig_crop = ig[:, y0:y0 + crop_h, x0:x0 + crop_w]
                        meta["ignore_mask"] = F.interpolate(
                            ig_crop.unsqueeze(0).float(), size=(H, W), mode="nearest"
                        ).squeeze(0).bool()
                    else:
                        ig_crop = ig[y0:y0 + crop_h, x0:x0 + crop_w]
                        meta["ignore_mask"] = F.interpolate(
                            ig_crop.unsqueeze(0).unsqueeze(0).float(), size=(H, W),
                            mode="nearest"
                        ).squeeze(0).squeeze(0).bool()

                return img_out, mask_out, meta

        return img, mask, meta  # fallback: no crop


# ── radiometric / noise transforms ────────────────────────────────────────────

class RandomSpeckleNoise(_Transform):
    """
    Simulate multiplicative speckle noise, the dominant noise model for
    Sentinel-1 GRD backscatter.

    For a single-look complex product with L looks, speckle follows a Gamma
    distribution: I_noisy = I * G, where G ~ Gamma(L, 1/L).  We approximate
    this with a log-normal draw:

        noise = exp( N(0, sigma²) )

    so that in log space this becomes additive Gaussian noise, consistent with
    the log-transform applied by SARDataset.

    Parameters
    ----------
    sigma : float
        Standard deviation of the log-normal noise.  At sigma=0.1 (default),
        the coefficient of variation is ~10 %, matching 4–9 look Sentinel-1
        GRD products.
    p : float
        Probability of applying the transform.
    """

    def __init__(self, sigma: float = 0.1, p: float = 0.5) -> None:
        super().__init__(p)
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0; got {sigma}")
        self.sigma = sigma

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta
        # In log space (which SARDataset applies), multiplicative speckle is
        # equivalent to additive Gaussian noise.
        noise = torch.randn_like(img) * self.sigma
        return img + noise, mask, meta


class RandomAdditiveGaussianNoise(_Transform):
    """
    Add small-amplitude Gaussian noise to simulate thermal noise floor and
    minor quantisation artefacts.

    Parameters
    ----------
    std_range : tuple of float
        Range (min, max) for the per-sample noise std drawn uniformly.
        Default: (0.0, 0.05).
    p : float
        Probability of applying the transform.
    """

    def __init__(
        self, std_range: Tuple[float, float] = (0.0, 0.05), p: float = 0.3
    ) -> None:
        super().__init__(p)
        self.std_range = std_range

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(img) * std
        return img + noise, mask, meta


class RandomBrightnessContrast(_Transform):
    """
    Per-sample brightness (additive shift) and contrast (multiplicative scale)
    jitter applied in normalised feature space.

    For SAR data this models incidence-angle variation across the swath and
    slight differences in scene reflectivity between acquisition dates.

    Parameters
    ----------
    brightness : float
        Maximum absolute brightness shift.  Default: 0.1.
    contrast : float
        Maximum contrast multiplicative deviation from 1.  Default: 0.1.
        A value of 0.1 means contrast ∈ [0.9, 1.1].
    p : float
        Probability of applying the transform.
    """

    def __init__(
        self, brightness: float = 0.1, contrast: float = 0.1, p: float = 0.5
    ) -> None:
        super().__init__(p)
        self.brightness = brightness
        self.contrast = contrast

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta
        b = random.uniform(-self.brightness, self.brightness)
        c = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        img = img * c + b
        return img, mask, meta


class RandomChannelDrop(_Transform):
    """
    Zero-out a random subset of channels to simulate partial-band missing data
    and encourage the model not to over-rely on any single polarisation.

    For dual-pol SAR (VV + VH), dropping one channel at a time is sensible.
    For fused SAR+optical, this also acts as a dropout on the optical bands,
    matching the ``optical_missing_policy="zeros"`` scenario seen at inference.

    Parameters
    ----------
    max_drop : int
        Maximum number of channels to zero out (default: 1).
    p : float
        Probability of applying the transform.
    """

    def __init__(self, max_drop: int = 1, p: float = 0.15) -> None:
        super().__init__(p)
        if max_drop < 1:
            raise ValueError(f"max_drop must be >= 1; got {max_drop}")
        self.max_drop = max_drop

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta
        n_channels = img.shape[0]
        n_drop = random.randint(1, min(self.max_drop, n_channels - 1))
        drop_indices = random.sample(range(n_channels), n_drop)
        img = img.clone()
        for idx in drop_indices:
            img[idx] = 0.0
        return img, mask, meta


class RandomGamma(_Transform):
    """
    Apply a random power-law (gamma) transformation to the image intensities.

    This is common in optical remote sensing to simulate atmospheric
    scattering variation.  For SAR it is a mild non-linearity that can model
    slight calibration differences between scenes.

    The transform is applied after normalisation, so values may extend beyond
    [0, 1].  Clamping is applied to keep the input in a valid range before
    the power is taken (avoids NaN from fractional powers of negative numbers).

    Parameters
    ----------
    gamma_range : tuple of float
        Range from which gamma is sampled uniformly.  Default: (0.8, 1.2).
    p : float
        Probability of applying the transform.
    """

    def __init__(
        self, gamma_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.3
    ) -> None:
        super().__init__(p)
        self.gamma_range = gamma_range

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta
        gamma = random.uniform(*self.gamma_range)
        # Shift to [0, ∞) before power (use offset so negative z-scores are handled)
        img_shifted = img - img.min()
        img_gamma = img_shifted.clamp(min=1e-8).pow(gamma)
        # Shift back so mean is preserved approximately
        img_out = img_gamma - img_gamma.mean() + img.mean()
        return img_out, mask, meta


class RandomCutout(_Transform):
    """
    Zero out a random rectangular region of the image (Cutout / DeVries 2017).

    For SAR flood segmentation with a tiny dataset this acts as a strong
    regulariser: the model cannot rely on any single spatial region.

    The valid_mask is updated so the zeroed region is excluded from loss —
    this prevents the model from being penalised for predictions in an
    artificially blank region.

    Parameters
    ----------
    max_frac : float
        Maximum width/height of the cutout as a fraction of image size.
        Default: 0.3 (30 %).
    fill_value : float
        Value used to fill the cutout region.  Default: 0.0.
    p : float
        Probability of applying the transform.
    """

    def __init__(
        self, max_frac: float = 0.3, fill_value: float = 0.0, p: float = 0.3
    ) -> None:
        super().__init__(p)
        self.max_frac = max_frac
        self.fill_value = fill_value

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if not self._should_apply():
            return img, mask, meta

        _, H, W = img.shape
        cut_h = int(random.uniform(0.05, self.max_frac) * H)
        cut_w = int(random.uniform(0.05, self.max_frac) * W)
        y0 = random.randint(0, H - cut_h)
        x0 = random.randint(0, W - cut_w)

        img = img.clone()
        img[:, y0:y0 + cut_h, x0:x0 + cut_w] = self.fill_value

        meta = dict(meta)
        vm = meta.get("valid_mask")
        if vm is not None:
            vm = vm.clone()
            vm[y0:y0 + cut_h, x0:x0 + cut_w] = False
            meta["valid_mask"] = vm

        return img, mask, meta


# ── weak-label-specific transforms ───────────────────────────────────────────

class WeakLabelBoundaryErosion(_Transform):
    """
    Erode the boundary pixels of weak (Otsu-derived) flood masks.

    Otsu threshold masks often have unreliable boundary pixels where water /
    non-water probability is ambiguous.  By eroding the mask boundary and
    marking eroded pixels as ignored (via ignore_mask), we shield the model
    from noisy label gradients in uncertain regions.

    This transform is a *no-op* when ``mask is None`` or when the
    ``ignore_mask`` key is absent from ``meta`` and ``create_ignore=False``.

    Parameters
    ----------
    erosion_px : int
        Number of pixels to erode from mask boundaries.  Default: 3.
    create_ignore : bool
        If True, create or extend ``meta["ignore_mask"]`` to cover eroded
        pixels.  If False, only the mask itself is modified.  Default: True.
    p : float
        Probability of applying the transform.  Default: 1.0 (always applied
        for weak labels — caller should gate on mode=="weak").
    """

    def __init__(
        self, erosion_px: int = 3, create_ignore: bool = True, p: float = 1.0
    ) -> None:
        super().__init__(p)
        if erosion_px < 1:
            raise ValueError(f"erosion_px must be >= 1; got {erosion_px}")
        self.erosion_px = erosion_px
        self.create_ignore = create_ignore

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if mask is None or not self._should_apply():
            return img, mask, meta

        # Identify boundary pixels via max-pool then subtract eroded mask
        mask_f = mask.float().unsqueeze(0)  # [1, 1, H, W]
        k = 2 * self.erosion_px + 1
        # dilated - eroded gives the boundary band
        dilated = F.max_pool2d(mask_f, kernel_size=k, stride=1, padding=self.erosion_px)
        eroded = -F.max_pool2d(-mask_f, kernel_size=k, stride=1, padding=self.erosion_px)
        boundary = (dilated - eroded).squeeze(0).squeeze(0)  # [H, W]
        boundary_bool = boundary > 0.5  # True = uncertain boundary pixel

        # Erode the mask: remove the boundary ring from flood region
        mask_new = eroded.squeeze(0).to(mask.dtype)  # [1, H, W]

        meta = dict(meta)
        if self.create_ignore:
            existing_ig = meta.get("ignore_mask")
            if existing_ig is not None:
                if not isinstance(existing_ig, torch.Tensor):
                    existing_ig = torch.as_tensor(existing_ig)
                if existing_ig.ndim == 3:
                    existing_ig = existing_ig.squeeze(0)
                new_ig = existing_ig | boundary_bool
            else:
                new_ig = boundary_bool
            meta["ignore_mask"] = new_ig.unsqueeze(0)  # [1, H, W]

        return img, mask_new, meta


# ── composition ───────────────────────────────────────────────────────────────

class Compose:
    """
    Sequentially apply a list of transforms.

    Each transform must accept and return ``(img, mask, meta)``.

    Parameters
    ----------
    transforms : list of callables
        Ordered list of transform objects (each with a ``p`` attribute handled
        internally).
    """

    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        for t in self.transforms:
            img, mask, meta = t(img, mask, meta)
        return img, mask, meta

    def __repr__(self) -> str:
        lines = ["Compose(["]
        for t in self.transforms:
            lines.append(f"    {t!r},")
        lines.append("])")
        return "\n".join(lines)


# ── factory functions ─────────────────────────────────────────────────────────

def build_train_transforms(
    modality: str = "sar",
    weak_label: bool = False,
    weak_erosion_px: int = 3,
    speckle_sigma: float = 0.1,
    brightness: float = 0.1,
    contrast: float = 0.1,
    max_frac_cutout: float = 0.3,
    max_drop: int = 1,
) -> Compose:
    """
    Build a sensible default training augmentation pipeline.

    Parameters
    ----------
    modality : {"sar", "optical", "fused"}
        Controls which radiometric transforms are included.
    weak_label : bool
        If True, prepend ``WeakLabelBoundaryErosion`` to shield the model
        from noisy Otsu-label boundary pixels.
    weak_erosion_px : int
        Erosion kernel radius passed to ``WeakLabelBoundaryErosion``.
    speckle_sigma : float
        Sigma for ``RandomSpeckleNoise``.  Only applied for SAR / fused data.
    brightness : float
        Brightness jitter range for ``RandomBrightnessContrast``.
    contrast : float
        Contrast jitter range for ``RandomBrightnessContrast``.
    max_frac_cutout : float
        Maximum cutout fraction of image size.
    max_drop : int
        Maximum number of channels to zero-drop.

    Returns
    -------
    Compose
        A composed transform callable compatible with SARDataset / OpticalDataset.

    Augmentation rationale
    ----------------------
    The Indian-subcontinent Sen1Floods11 subset has ~138 hand-labeled tiles and
    ~446 weakly-labeled tiles (Otsu threshold).  After 20 % overlap patching
    these expand to ~3 k–11 k patches, but the effective variety is still low
    given the narrow geography and single sensor.  The pipeline below applies:

    * **Flips** (H + V): free-lunch — flood extent is orientation-agnostic.
    * **90° rotations**: expand geometric diversity; important for river-bend
      patches where the flood direction encodes orientation.
    * **Fine rotation (±15°)**: further break orientation bias.
    * **Random crop+resize**: simulate scale variation (pond vs. river inundation).
    * **Speckle noise** (SAR/fused only): models look-number variation; applied
      additively in log space, consistent with log_transform=True in SARDataset.
    * **Brightness/contrast**: models incidence-angle / calibration variation.
    * **Gamma** (optical only): atmospheric scattering variation.
    * **Channel drop**: encourages the model to be robust to missing polarisation
      (SAR) or clouded optical bands (fused).
    * **Cutout**: strong regulariser for the small dataset regime; updates
      valid_mask so cutout regions are excluded from loss.
    * **Weak-label erosion** (weak_label=True): removes noisy boundary labels
      from Otsu masks and marks them as ignored.
    """
    transforms: List[Any] = []

    # Weak-label boundary erosion comes first so downstream spatial augmentations
    # propagate the updated ignore_mask correctly.
    if weak_label:
        transforms.append(
            WeakLabelBoundaryErosion(erosion_px=weak_erosion_px, create_ignore=True, p=1.0)
        )

    # ── geometric ──────────────────────────────────────────────────────────
    transforms += [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(p=0.5),
        RandomRotationFine(max_angle=15.0, p=0.3),
        RandomCropAndResize(scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.5),
    ]

    # ── radiometric ────────────────────────────────────────────────────────
    if modality in {"sar", "fused"}:
        transforms.append(RandomSpeckleNoise(sigma=speckle_sigma, p=0.5))
    if modality in {"optical", "fused"}:
        transforms.append(RandomGamma(gamma_range=(0.8, 1.2), p=0.3))

    transforms += [
        RandomBrightnessContrast(brightness=brightness, contrast=contrast, p=0.5),
        RandomAdditiveGaussianNoise(std_range=(0.0, 0.05), p=0.3),
    ]

    # ── structural ─────────────────────────────────────────────────────────
    if modality in {"sar", "fused"}:
        transforms.append(RandomChannelDrop(max_drop=max_drop, p=0.15))

    transforms.append(RandomCutout(max_frac=max_frac_cutout, fill_value=0.0, p=0.3))

    return Compose(transforms)


def build_val_transforms() -> Compose:
    """
    Return an empty (identity) transform pipeline for validation / inference.

    No augmentation is applied; the function exists so that dataset
    construction code can call it symmetrically with ``build_train_transforms``
    without needing a None check.
    """
    return Compose([])
