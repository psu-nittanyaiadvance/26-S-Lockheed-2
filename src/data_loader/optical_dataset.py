"""
optical_dataset.py  –  Sentinel-2 optical dataset for flood segmentation.

Design notes
------------
* Mirrors SARDataset's public interface so it can be used as a drop-in
  replacement or alongside SARDataset in FusedDataset.
* S2 reflectance values are expected as either:
    - Integer DN in [0, 10000]  (ESA SAFE / GEE default export)
    - Float reflectance in [0.0, 1.0]  (some GEE pipelines divide by 10000)
  The loader auto-detects which range is present and normalises to [0, 1].
* Cloud / SCL masking: if a cloud-mask band path is provided (or a dedicated
  SCL band index is specified inside the image stack), those pixels are
  flagged in valid_mask so they are excluded from loss computation.
* log_transform is intentionally NOT supported for optical data.
* Band selection via `s2_bands` lets you pick a subset of the stack (e.g.
  [0,1,2,3] for B2/B3/B4/B8 only).

Supported normalisation types (same as SARDataset):
    "none"   – raw [0,1] reflectance floats
    "zscore" – (x - mean) / std, stats provided by compute_running_mean_std
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import torch
from torch.utils.data import Dataset

from .sar_dataset import (   # re-use shared helpers
    NormalizeCfg,
    Sample,
    _infer_ids_are_paths,
    _is_tif_name,
)


# ── constants ────────────────────────────────────────────────────────────────

# Sentinel-2 L2A surface reflectance scale factor (ESA / GEE default export).
_S2_DN_SCALE: float = 10_000.0

# If the max value in the image is above this threshold we assume DN [0,10000]
# and divide by _S2_DN_SCALE; otherwise we assume already-normalised [0,1].
_DN_THRESHOLD: float = 5.0


# ── OpticalDataset ───────────────────────────────────────────────────────────

class OpticalDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 optical tiles and optional masks.

    Parameters
    ----------
    img_root : Path or None
        Directory containing S2 .tif tiles (required when ids_or_paths are IDs).
    mask_root : Path or None
        Directory containing binary flood mask .tif files.
        Required when mode != "none".
    ids_or_paths : sequence of str or Path
        Either tile IDs (stems) or full .tif paths.
    mode : {"strong", "weak", "none"}
        Whether to load ground-truth masks.
    s2_bands : sequence of int or None
        Zero-based band indices to load.  None = all bands.
        Example: [0, 1, 2, 3] loads the first 4 bands only.
    cloud_mask_root : Path or None
        Optional directory of cloud/SCL mask .tif tiles (same stems as img_root).
        Where cloud pixels are flagged, valid_mask is set to False.
    cloud_mask_invalid_values : sequence of int
        SCL class values to treat as invalid (cloudy/shadow).
        Defaults to ESA L2A SCL classes 3,8,9,10,11
        (cloud shadow, medium/high cloud probability, cirrus, snow/ice).
    reflectance_clip_percentile : float
        Per-band percentile used for robust clipping before normalisation.
        Set to 0.0 to disable clipping entirely.  Default: 2.0 (2–98 %).
    transforms : callable or None
        Callable(img_tensor, mask_tensor, metadata) → same types.
    normalize_cfg : str, dict, or None
        "none", or {"type": "zscore", "mean": ..., "std": ...}.
    validate : bool
        If True, files are opened at init and bad samples are dropped.
    ids_are_paths : bool or None
        Auto-inferred if None.
    mask_id_suffix_map : dict or None
        Maps image ID suffixes → mask ID suffixes (same as SARDataset).
    expected_img_bands : int or None
        Hard-check the number of bands after s2_bands selection.
    """

    def __init__(
        self,
        img_root: Optional[Union[str, Path]],
        mask_root: Optional[Union[str, Path]],
        ids_or_paths: Sequence[Union[str, Path]],
        mode: str,
        s2_bands: Optional[Sequence[int]] = None,
        cloud_mask_root: Optional[Union[str, Path]] = None,
        cloud_mask_invalid_values: Sequence[int] = (3, 8, 9, 10, 11),
        reflectance_clip_percentile: float = 2.0,
        transforms: Optional[Callable[..., Any]] = None,
        normalize_cfg: NormalizeCfg = None,
        validate: bool = False,
        ids_are_paths: Optional[bool] = None,
        mask_id_suffix_map: Optional[Dict[str, str]] = None,
        expected_img_bands: Optional[int] = None,
    ) -> None:
        if mode not in {"weak", "strong", "none"}:
            raise ValueError(f"mode must be 'weak', 'strong', or 'none'; got {mode!r}")
        if not ids_or_paths:
            raise ValueError("ids_or_paths is empty")
        if not (0.0 <= reflectance_clip_percentile < 50.0):
            raise ValueError(
                "reflectance_clip_percentile must be in [0, 50); "
                f"got {reflectance_clip_percentile}"
            )

        self.mode = mode
        self.s2_bands: Optional[List[int]] = list(s2_bands) if s2_bands is not None else None
        self.cloud_mask_root = Path(cloud_mask_root) if cloud_mask_root is not None else None
        self.cloud_mask_invalid_values = set(int(v) for v in cloud_mask_invalid_values)
        self.reflectance_clip_percentile = float(reflectance_clip_percentile)
        self.transforms = transforms
        self.mask_id_suffix_map = dict(mask_id_suffix_map) if mask_id_suffix_map else None
        self.expected_img_bands = expected_img_bands
        if self.expected_img_bands is not None and self.expected_img_bands <= 0:
            raise ValueError("expected_img_bands must be a positive integer")

        if ids_are_paths is None:
            ids_are_paths = _infer_ids_are_paths(ids_or_paths)
        self.ids_are_paths = bool(ids_are_paths)

        self.img_root = Path(img_root) if img_root is not None else None
        self.mask_root = Path(mask_root) if mask_root is not None else None

        if not self.ids_are_paths:
            if self.img_root is None:
                raise ValueError("img_root is required when ids_or_paths are IDs")
            if not self.img_root.is_dir():
                raise ValueError(
                    f"img_root does not exist or is not a directory: {self.img_root}"
                )

        if self.mode != "none":
            if self.mask_root is None:
                raise ValueError("mask_root is required for mode='weak' or mode='strong'")
            if not self.mask_root.is_dir():
                raise ValueError(
                    f"mask_root does not exist or is not a directory: {self.mask_root}"
                )

        if self.cloud_mask_root is not None and not self.cloud_mask_root.is_dir():
            raise ValueError(
                f"cloud_mask_root does not exist or is not a directory: {self.cloud_mask_root}"
            )

        self._normalize_type: str = "none"
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._parse_normalize_cfg(normalize_cfg)

        self.samples: List[Sample] = self._build_samples(ids_or_paths)
        if validate:
            self.samples = self._validate_samples(self.samples)
            if not self.samples:
                raise ValueError("No valid samples remain after validation")

    # ── normalisation ────────────────────────────────────────────────────────

    def _parse_normalize_cfg(self, normalize_cfg: NormalizeCfg) -> None:
        """Identical logic to SARDataset._parse_normalize_cfg."""
        if normalize_cfg is None:
            self._normalize_type = "none"
            return
        if isinstance(normalize_cfg, str):
            self._normalize_type = normalize_cfg.lower()
        elif isinstance(normalize_cfg, dict):
            self._normalize_type = str(normalize_cfg.get("type", "none")).lower()
            if self._normalize_type == "zscore":
                if "mean" not in normalize_cfg or "std" not in normalize_cfg:
                    raise ValueError("normalize_cfg for 'zscore' must include 'mean' and 'std'")
                self._norm_mean = np.asarray(normalize_cfg["mean"], dtype=np.float32)
                self._norm_std = np.asarray(normalize_cfg["std"], dtype=np.float32)
        else:
            raise TypeError("normalize_cfg must be None, str, or dict")
        if self._normalize_type not in {"none", "zscore"}:
            raise ValueError(
                f"normalize_cfg type must be 'none' or 'zscore', got {self._normalize_type!r}"
            )

    def _apply_normalization(self, img: np.ndarray) -> np.ndarray:
        if self._normalize_type == "none":
            return img
        if self._normalize_type == "zscore":
            if self._norm_mean is None or self._norm_std is None:
                raise ValueError("zscore normalization requires 'mean' and 'std'")
            c = img.shape[0]
            if self._norm_mean.size != c or self._norm_std.size != c:
                raise ValueError(
                    f"Normalization stats length mismatch: "
                    f"mean/std length {self._norm_mean.size}/{self._norm_std.size}, expected {c}"
                )
            if np.any(self._norm_std <= 0):
                raise ValueError("Normalization std must be > 0 for all bands")
            mean = self._norm_mean.reshape(c, 1, 1)
            std = self._norm_std.reshape(c, 1, 1)
            return (img - mean) / std
        raise ValueError(f"Unknown normalization type: {self._normalize_type}")

    # ── path resolution ──────────────────────────────────────────────────────

    def _resolve_raster_path(self, root: Path, name_or_id: str) -> Path:
        p = Path(name_or_id)
        if _is_tif_name(name_or_id):
            return root / p.name
        tif = root / f"{name_or_id}.tif"
        tiff = root / f"{name_or_id}.tiff"
        if tif.exists():
            return tif
        if tiff.exists():
            return tiff
        return tif

    def _resolve_mask_id(self, sample_id: str) -> str:
        if self.mask_root is not None:
            direct_tif = self.mask_root / f"{sample_id}.tif"
            direct_tiff = self.mask_root / f"{sample_id}.tiff"
            if direct_tif.exists() or direct_tiff.exists():
                return sample_id
        if not self.mask_id_suffix_map:
            return sample_id
        for img_suffix, mask_suffix in self.mask_id_suffix_map.items():
            if sample_id.endswith(img_suffix):
                return f"{sample_id[:-len(img_suffix)]}{mask_suffix}"
        return sample_id

    # ── sample building ──────────────────────────────────────────────────────

    def _build_samples(self, ids_or_paths: Sequence[Union[str, Path]]) -> List[Sample]:
        samples: List[Sample] = []
        for item in ids_or_paths:
            s = str(item)
            if self.ids_are_paths:
                img_path = Path(s)
                sample_id = img_path.stem
            else:
                sample_id = Path(s).stem if _is_tif_name(s) else s
                if self.img_root is None:
                    raise ValueError("img_root is required when ids_or_paths are IDs")
                img_path = self._resolve_raster_path(self.img_root, s)

            mask_path: Optional[Path] = None
            if self.mode != "none":
                if self.mask_root is None:
                    raise ValueError("mask_root is required for labeled modes")
                mask_id = self._resolve_mask_id(sample_id)
                mask_path = self._resolve_raster_path(self.mask_root, mask_id)

            samples.append(
                {
                    "id": sample_id,
                    "img_path": str(img_path),
                    "mask_path": str(mask_path) if mask_path is not None else None,
                }
            )
        return samples

    def _validate_samples(self, samples: List[Sample]) -> List[Sample]:
        valid: List[Sample] = []
        for sample in samples:
            img_path = sample["img_path"]
            mask_path = sample["mask_path"]
            sample_id = sample["id"]
            try:
                with rasterio.open(img_path) as src:
                    img_h, img_w = src.height, src.width
                    if src.count < 1:
                        raise ValueError(f"Image has zero bands (id='{sample_id}')")

                if self.mode != "none":
                    if mask_path is None:
                        raise ValueError(
                            f"Missing mask path for labeled mode (id='{sample_id}')"
                        )
                    with rasterio.open(mask_path) as msrc:
                        mask_h, mask_w = msrc.height, msrc.width
                    if (img_h, img_w) != (mask_h, mask_w):
                        raise ValueError(
                            f"Mask/image shape mismatch for id '{sample_id}': "
                            f"mask {(mask_h, mask_w)} vs image {(img_h, img_w)}"
                        )

                valid.append(sample)
            except Exception as e:
                warnings.warn(f"Dropping sample '{sample_id}' due to validation error: {e}")
        return valid

    # ── image loading ────────────────────────────────────────────────────────

    def _load_image(
        self, img_path: str, sample_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a Sentinel-2 image, returning ``(img, valid_mask)``.

        Processing pipeline
        -------------------
        1. Read all (or selected) bands via rasterio.
        2. Auto-detect DN vs reflectance and scale to [0, 1].
        3. Flag pixels that are NaN / Inf or exactly 0 in all bands
           (common nodata sentinel in S2 exports) in ``invalid``.
        4. Optionally apply per-band percentile clipping (robust to outliers).
        5. Optionally load SCL / cloud mask and extend invalid mask.
        6. Band-min impute remaining NaN/Inf so tensor is fully finite.
        7. Apply normalisation (zscore or none).

        Returns
        -------
        img : np.ndarray, shape (C, H, W), dtype float32, fully finite.
        valid_mask : np.ndarray, shape (H, W), dtype bool.
            False wherever any band was invalid or cloud-masked.
        """
        # ---- 1. Read bands --------------------------------------------------
        try:
            with rasterio.open(img_path) as src:
                total_bands = src.count
                if self.expected_img_bands is not None:
                    expected = (
                        self.expected_img_bands
                        if self.s2_bands is None
                        else self.expected_img_bands
                    )
                if self.s2_bands is not None:
                    out_of_range = [b for b in self.s2_bands if b >= total_bands or b < 0]
                    if out_of_range:
                        raise ValueError(
                            f"s2_bands indices {out_of_range} out of range for image with "
                            f"{total_bands} bands (id='{sample_id}')"
                        )
                    # rasterio bands are 1-indexed
                    rasterio_bands = [b + 1 for b in self.s2_bands]
                    img = src.read(rasterio_bands)  # (C_sel, H, W)
                else:
                    img = src.read()  # (C, H, W)
        except RasterioIOError as e:
            raise RuntimeError(
                f"Failed to read image for id '{sample_id}' at '{img_path}': {e}"
            ) from e

        if self.expected_img_bands is not None and img.shape[0] != self.expected_img_bands:
            raise ValueError(
                f"Expected {self.expected_img_bands} bands after selection for id "
                f"'{sample_id}', got {img.shape[0]}"
            )

        img = img.astype(np.float32, copy=False)

        # ---- 2. Scale to [0, 1] reflectance ---------------------------------
        # Use the 99th-percentile of finite values to detect DN range.
        finite_vals = img[np.isfinite(img)]
        if finite_vals.size > 0 and np.nanpercentile(finite_vals, 99) > _DN_THRESHOLD:
            img = img / _S2_DN_SCALE

        # ---- 3. Flag invalid pixels -----------------------------------------
        # NaN / Inf from the file
        invalid = ~np.isfinite(img)  # (C, H, W), bool

        # All-zero pixels across all bands = common nodata sentinel in S2 exports
        all_zero = (img == 0.0).all(axis=0)  # (H, W)
        invalid |= all_zero[np.newaxis, :, :]

        # Values outside physical range [0, 1] after scaling are suspicious;
        # clamp rather than flag, as some surface types can slightly exceed 1.0.
        img = np.clip(img, 0.0, 1.0)

        # ---- 4. Percentile clipping (per band, robust to outliers) ----------
        if self.reflectance_clip_percentile > 0.0:
            lo_pct = self.reflectance_clip_percentile
            hi_pct = 100.0 - lo_pct
            img_t = torch.from_numpy(img)
            flat = img_t.view(img_t.shape[0], -1)
            lo = torch.nanquantile(flat, lo_pct / 100.0, dim=1).view(-1, 1, 1)
            hi = torch.nanquantile(flat, hi_pct / 100.0, dim=1).view(-1, 1, 1)
            img_t = torch.clamp(img_t, lo, hi)
            img = img_t.numpy()

        # ---- 5. Cloud / SCL masking -----------------------------------------
        if self.cloud_mask_root is not None:
            cloud_path = self._resolve_raster_path(
                self.cloud_mask_root, Path(img_path).stem
            )
            if cloud_path.exists():
                try:
                    with rasterio.open(str(cloud_path)) as csrc:
                        scl = csrc.read(1)  # (H, W), uint8/uint16
                    cloud_invalid = np.isin(scl, list(self.cloud_mask_invalid_values))
                    # Broadcast cloud mask across all bands
                    invalid |= cloud_invalid[np.newaxis, :, :]
                except Exception as e:
                    warnings.warn(
                        f"Could not read cloud mask for id '{sample_id}' at "
                        f"'{cloud_path}': {e}. Cloud masking skipped for this tile."
                    )
            else:
                warnings.warn(
                    f"Cloud mask not found for id '{sample_id}' at '{cloud_path}'. "
                    "Cloud masking skipped for this tile."
                )

        # ---- 6. valid_mask: True where ALL bands are valid ------------------
        valid_mask = ~invalid.any(axis=0)  # (H, W), bool

        # ---- 7. Band-min impute so tensor stays fully finite ----------------
        if not valid_mask.all():
            for c in range(img.shape[0]):
                band = img[c]
                finite_vals = band[np.isfinite(band)]
                fill = float(finite_vals.min()) if finite_vals.size > 0 else 0.0
                band[~np.isfinite(band)] = fill
                img[c] = band

        img = self._apply_normalization(img)
        return img, valid_mask

    def _load_mask(
        self, mask_path: str, expected_hw: Tuple[int, int], sample_id: str
    ) -> np.ndarray:
        """Identical to SARDataset._load_mask; kept here for self-containment."""
        try:
            with rasterio.open(mask_path) as src:
                if src.count < 1:
                    raise ValueError("Mask has zero bands")
                mask = src.read(1)
        except RasterioIOError as e:
            raise RuntimeError(
                f"Failed to read mask for id '{sample_id}' at '{mask_path}': {e}"
            ) from e
        if mask.shape != expected_hw:
            raise ValueError(
                f"Mask shape mismatch for id '{sample_id}': "
                f"mask {mask.shape}, image {expected_hw}"
            )
        return (mask > 0).astype(np.uint8)[None, :, :]

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        sample = self.samples[idx]
        sample_id = sample["id"]
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        img, valid_mask = self._load_image(img_path, sample_id)
        if not np.isfinite(img).all():
            raise ValueError(
                f"Optical image contains NaN/Inf after preprocessing for id '{sample_id}'"
            )
        if valid_mask.ndim != 2:
            raise ValueError(
                f"valid_mask must have shape [H, W] for id '{sample_id}', got {valid_mask.shape}"
            )

        img_tensor = torch.from_numpy(img).to(dtype=torch.float32)
        valid_mask_tensor = torch.from_numpy(valid_mask).to(dtype=torch.bool)  # [H, W]

        mask_tensor: Optional[torch.Tensor] = None
        if self.mode != "none":
            if mask_path is None:
                raise RuntimeError(f"Missing mask path for id '{sample_id}'")
            mask_np = self._load_mask(mask_path, (img.shape[1], img.shape[2]), sample_id)
            mask_tensor = torch.from_numpy(mask_np).to(dtype=torch.uint8)

        metadata: Dict[str, Any] = {
            "id": sample_id,
            "img_path": img_path,
            "mask_path": mask_path,
            "valid_mask": valid_mask_tensor,
            "modality": "optical",
        }

        if self.transforms is not None:
            out = self.transforms(img_tensor, mask_tensor, metadata)
            if isinstance(out, tuple) and len(out) == 3:
                img_tensor, mask_tensor, metadata = out
            elif isinstance(out, tuple) and len(out) == 2:
                img_tensor, mask_tensor = out
            else:
                raise ValueError(
                    "transforms must return (image, mask, metadata) or (image, mask)"
                )

        return img_tensor, mask_tensor, metadata

    def __repr__(self) -> str:
        bands = f"bands={self.s2_bands}" if self.s2_bands is not None else "bands=all"
        return (
            f"OpticalDataset("
            f"n={len(self.samples)}, "
            f"{bands}, "
            f"mode={self.mode!r}, "
            f"normalize={self.normalize_type!r}, "
            f"cloud_mask={'yes' if self.cloud_mask_root else 'no'})"
        )

    @property
    def normalize_type(self) -> str:
        return self._normalize_type
