from __future__ import annotations

import csv
import hashlib
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
from numpy.strings import lower
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.errors import RasterioIOError


class Sample(TypedDict):
    id: str
    img_path: str
    mask_path: Optional[str]


NormalizeCfg = Union[str, Dict[str, Any], None]
DEFAULT_QUANTILE_CACHE_ROOT = Path("cache") / "quantiles"
_QUANTILE_CACHE_VERSION = 1


def _is_tif_name(name: str) -> bool:
    ext = Path(name).suffix.lower()
    return ext in (".tif", ".tiff")


def _infer_ids_are_paths(items: Sequence[Union[str, Path]]) -> bool:
    for item in items:
        s = str(item)
        p = Path(s)
        if p.is_absolute():
            return True
        if _is_tif_name(s):
            return True
        if os.sep in s or (os.altsep and os.altsep in s):
            return True
    return False


def _infer_split_from_path(path: str) -> str:
    parts = Path(path).parts
    for token in ("WeaklyLabeled", "HandLabeled"):
        if token in parts:
            return token
    return "all"


def _quantile_cache_path(
    cache_root: Optional[Path],
    dataset_type: str,
    sample_id: str,
    img_path: str,
) -> Optional[Path]:
    if cache_root is None:
        return None
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sample_id)
    digest = hashlib.sha1(str(Path(img_path).resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_root / dataset_type / f"{safe_id}_{digest}.pt"


def _file_cache_fingerprint(img_path: str) -> Dict[str, Any]:
    path = Path(img_path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _load_quantile_cache(
    cache_path: Optional[Path],
    expected_key: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        cached = torch.load(cache_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(cached, dict) or cached.get("key") != expected_key:
        return None
    lower = cached.get("lower")
    upper = cached.get("upper")
    if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
        return None
    return cached


def _save_quantile_cache(
    cache_path: Optional[Path],
    key: Dict[str, Any],
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "key": key,
            "lower": lower.detach().cpu(),
            "upper": upper.detach().cpu(),
        }
        if extra:
            payload.update(extra)
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.{os.getpid()}.tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        warnings.warn(f"Could not write quantile cache '{cache_path}': {exc}")


class SARDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 SAR tiles and optional masks.

    Directory structure assumptions:
    - Image tiles are .tif/.tiff readable by rasterio.
    - If ids_or_paths are IDs, image path is img_root/<id>.tif (fallback to .tiff).
    - If ids_or_paths are paths, use them directly; mask path is mask_root/<id>.tif.

    Mode behavior:
    - mode="weak": read masks from mask_root (weak label directory).
    - mode="strong": read masks from mask_root (hand label directory).
    - mode="none": no masks returned.

    ids_or_paths behavior:
    - IDs (e.g., "tile_000123"): resolved under img_root and mask_root.
    - Paths (absolute or containing separators, or .tif/.tiff suffix): used directly for images;
      masks are still resolved under mask_root using the image stem.

    Time-matched behavior (use_time_matched=True):
    - Reads an 8-band stack (VV, VH + 6x S2 bands) from time_matched_root/{split}/{id}.tif
      or uses the time_matched_manifest if present.
    - Metadata always includes time_matched_status and time_matched_path (even if None).

    Missing policies for time-matched stacks:
    - "zeros": return an all-zero 8-band stack (warning emitted once).
    - "skip": drop samples without a valid time-matched stack at init time.
    - "raise": raise at __getitem__ if a time-matched stack is missing.

    The Dataset does not attempt to map weak<->strong IDs; it only uses the
    provided IDs or image paths.
    """

    def __init__(
        self,
        img_root: Optional[Union[str, Path]],
        mask_root: Optional[Union[str, Path]],
        ids_or_paths: Sequence[Union[str, Path]],
        mode: str,
        transforms: Optional[Callable[..., Any]] = None,
        normalize_cfg: NormalizeCfg = None,
        log_transform: bool = False,
        validate: bool = False,
        ids_are_paths: Optional[bool] = None,
        use_time_matched: bool = False,
        time_matched_root: Union[str, Path] = "data/derived/gee_time_matched",
        time_matched_manifest: Union[str, Path] = "data/derived/gee_time_matched_manifest.csv",
        time_matched_missing_policy: str = "zeros",
        mask_id_suffix_map: Optional[Dict[str, str]] = None,
        expected_img_bands: Optional[int] = None,
        quantile_cache_root: Optional[Union[str, Path]] = DEFAULT_QUANTILE_CACHE_ROOT,
        
    ) -> None:
        if mode not in {"weak", "strong", "none"}:
            raise ValueError(f"mode must be one of 'weak', 'strong', 'none'; got {mode}")

        if not ids_or_paths:
            raise ValueError("ids_or_paths is empty")

        self.mode = mode
        self.transforms = transforms
        self.log_transform = bool(log_transform)

        if ids_are_paths is None:
            ids_are_paths = _infer_ids_are_paths(ids_or_paths)
        self.ids_are_paths = bool(ids_are_paths)

        self.img_root = Path(img_root) if img_root is not None else None
        self.mask_root = Path(mask_root) if mask_root is not None else None
        self.mask_id_suffix_map = dict(mask_id_suffix_map) if mask_id_suffix_map else None
        self.expected_img_bands = expected_img_bands
        if self.expected_img_bands is not None and self.expected_img_bands <= 0:
            raise ValueError("expected_img_bands must be a positive integer")
        if os.environ.get("LOCKDOCKS_DISABLE_QUANTILE_CACHE") == "1":
            quantile_cache_root = None
        self.quantile_cache_root = (
            Path(quantile_cache_root) if quantile_cache_root is not None else None
        )
        self._profile_quantiles = os.environ.get("LOCKDOCKS_PROFILE_QUANTILES") == "1"
        self._quantile_profile: Dict[str, Any] = {
            "load_image_calls": 0,
            "load_image_seconds": 0.0,
            "nanquantile_calls": 0,
            "nanquantile_seconds": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "shapes": {},
        }

        if not self.ids_are_paths:
            if self.img_root is None:
                raise ValueError("img_root is required when ids_or_paths are IDs")
            if not self.img_root.is_dir():
                raise ValueError(f"img_root does not exist or is not a directory: {self.img_root}")

        if self.mode != "none":
            if self.mask_root is None:
                raise ValueError("mask_root is required for mode='weak' or mode='strong'")
            if not self.mask_root.is_dir():
                raise ValueError(f"mask_root does not exist or is not a directory: {self.mask_root}")

        self.use_time_matched = bool(use_time_matched)
        self.time_matched_root = Path(time_matched_root)
        self.time_matched_manifest = Path(time_matched_manifest)
        self.time_matched_missing_policy = time_matched_missing_policy
        if self.time_matched_missing_policy not in {"zeros", "skip", "raise"}:
            raise ValueError(
                "time_matched_missing_policy must be one of {'zeros','skip','raise'}"
            )
        self._tm_index: Dict[str, Dict[str, str]] = (
            self._load_time_matched_index() if self.use_time_matched else {}
        )
        self._warned_missing_tm = False

        self._normalize_type: str = "none"
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._parse_normalize_cfg(normalize_cfg)

        self.samples: List[Sample] = self._build_samples(ids_or_paths)
        if self.use_time_matched and self.time_matched_missing_policy == "skip":
            self.samples = [s for s in self.samples if self._time_matched_exists(s)]
            if not self.samples:
                raise ValueError("No samples remain after applying time-matched skip policy")
        if validate:
            self.samples = self._validate_samples(self.samples)
            if not self.samples:
                raise ValueError("No valid samples remain after validation")

    def _load_time_matched_index(self) -> Dict[str, Dict[str, str]]:
        if not self.time_matched_manifest.exists():
            warnings.warn(
                f"Time-matched manifest not found at {self.time_matched_manifest}. "
                "Will fall back to filesystem checks."
            )
            return {}
        index: Dict[str, Dict[str, str]] = {}
        with self.time_matched_manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "sample_id" not in row:
                    continue
                index[str(row["sample_id"])] = row
        return index

    def _time_matched_path(self, sample: Sample) -> Tuple[Optional[Path], Optional[str]]:
        sample_id = sample["id"]
        row = self._tm_index.get(sample_id)
        if row and row.get("status") == "ok" and row.get("path"):
            path = Path(row["path"])
            if path.exists():
                return path, "ok"
        split = _infer_split_from_path(sample["img_path"])
        path = self.time_matched_root / split / f"{sample_id}.tif"
        if path.exists():
            return path, row.get("status") if row else "ok"
        return None, row.get("status") if row else None

    def _time_matched_exists(self, sample: Sample) -> bool:
        path, status = self._time_matched_path(sample)
        if status and status != "ok":
            return False
        return path is not None and path.exists()

    def _parse_normalize_cfg(self, normalize_cfg: NormalizeCfg) -> None:
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
            raise ValueError(f"normalize_cfg type must be 'none' or 'zscore', got {self._normalize_type}")

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
                    raw = src.read().astype(np.float32, copy=False)

                # Warn on high NaN / non-positive pixel rates so users can
                # investigate or filter tiles before training.
                non_positive_frac = float((raw <= 0).mean())
                nan_frac = float(~np.isfinite(raw).mean())
                if nan_frac > 0.05:
                    warnings.warn(
                        f"Sample '{sample_id}' has {nan_frac:.1%} non-finite pixels — "
                        "consider excluding this tile."
                    )
                elif non_positive_frac > 0.10:
                    warnings.warn(
                        f"Sample '{sample_id}' has {non_positive_frac:.1%} non-positive "
                        "pixels (will become NaN after log-transform)."
                    )

                if self.mode != "none":
                    if mask_path is None:
                        raise ValueError(f"Missing mask path for labeled mode (id='{sample_id}')")
                    with rasterio.open(mask_path) as msrc:
                        mask_h, mask_w = msrc.height, msrc.width
                    if (img_h, img_w) != (mask_h, mask_w):
                        raise ValueError(
                            f"Mask shape mismatch for id '{sample_id}': "
                            f"mask {(mask_h, mask_w)} vs image {(img_h, img_w)}"
                        )

                valid.append(sample)
            except Exception as e:
                warnings.warn(f"Dropping sample '{sample['id']}' due to validation error: {e}")
        return valid

    def _apply_normalization(self, img: np.ndarray) -> np.ndarray:
        if self._normalize_type == "none":
            return img

        if self._normalize_type == "zscore":
            if self._norm_mean is None or self._norm_std is None:
                raise ValueError("normalize_cfg for 'zscore' must include 'mean' and 'std'")
            c = img.shape[0]
            if self._norm_mean.size != c or self._norm_std.size != c:
                raise ValueError(
                    f"Normalization stats length mismatch: got mean/std length "
                    f"{self._norm_mean.size}/{self._norm_std.size}, expected {c}"
                )
            if np.any(self._norm_std <= 0):
                raise ValueError("Normalization std must be > 0 for all bands")
            mean = self._norm_mean.reshape(c, 1, 1)
            std = self._norm_std.reshape(c, 1, 1)
            return (img - mean) / std

        raise ValueError(f"Unknown normalization type: {self._normalize_type}")

    def _record_quantile_shape(self, flat: torch.Tensor) -> None:
        if not self._profile_quantiles:
            return
        shapes = self._quantile_profile["shapes"]
        key = f"{tuple(flat.shape)} {flat.dtype} {flat.device}"
        shapes[key] = int(shapes.get(key, 0)) + 1

    def _record_nanquantile_time(self, flat: torch.Tensor, seconds: float) -> None:
        if not self._profile_quantiles:
            return
        self._quantile_profile["nanquantile_calls"] += 1
        self._quantile_profile["nanquantile_seconds"] += seconds
        self._record_quantile_shape(flat)

    def _record_cache_hit(self) -> None:
        if self._profile_quantiles:
            self._quantile_profile["cache_hits"] += 1

    def _record_cache_miss(self) -> None:
        if self._profile_quantiles:
            self._quantile_profile["cache_misses"] += 1

    def _record_load_image_time(self, started_at: Optional[float]) -> None:
        if started_at is None or not self._profile_quantiles:
            return
        self._quantile_profile["load_image_calls"] += 1
        self._quantile_profile["load_image_seconds"] += time.perf_counter() - started_at

    def quantile_profile_summary(self, *, reset: bool = False) -> Dict[str, Any]:
        """
        Return lightweight dataloader quantile counters.

        Enable collection with ``LOCKDOCKS_PROFILE_QUANTILES=1``. Call this at
        epoch boundaries if per-epoch counters are needed.
        """
        summary = {
            "load_image_calls": self._quantile_profile["load_image_calls"],
            "load_image_seconds": self._quantile_profile["load_image_seconds"],
            "nanquantile_calls": self._quantile_profile["nanquantile_calls"],
            "nanquantile_seconds": self._quantile_profile["nanquantile_seconds"],
            "cache_hits": self._quantile_profile["cache_hits"],
            "cache_misses": self._quantile_profile["cache_misses"],
            "shapes": dict(self._quantile_profile["shapes"]),
        }
        if reset:
            self._quantile_profile = {
                "load_image_calls": 0,
                "load_image_seconds": 0.0,
                "nanquantile_calls": 0,
                "nanquantile_seconds": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "shapes": {},
            }
        return summary

    def _load_image(
        self, img_path: str, sample_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a SAR image and return ``(img, valid_mask)``.

        ``valid_mask`` is a boolean array of shape ``(H, W)`` that is ``True``
        wherever *every* band contains a finite, valid value and ``False`` for
        any pixel that was NaN / Inf at any stage of the processing pipeline.
        This mask should be propagated to the loss function so that invalid
        pixels never contribute gradient.

        NaN handling strategy (in order):
        1. Pixels that are already NaN/Inf after rasterio read are flagged.
        2. Before log-transform, non-positive values (which would produce
           ``-inf`` / ``NaN``) are replaced with ``NaN`` so they are caught by
           the same mask rather than silently producing bad values.
        3. After all processing, any pixel still non-finite is imputed with the
           per-band minimum of *valid* pixels (band-min imputation).  This keeps
           the tensor fully finite for convolution ops, but the accompanying
           ``valid_mask`` ensures these pixels are excluded from the loss.
        """
        load_started_at = time.perf_counter() if self._profile_quantiles else None
        try:
            with rasterio.open(img_path) as src:
                if self.expected_img_bands is not None and src.count != self.expected_img_bands:
                    raise ValueError(
                        f"Expected {self.expected_img_bands} image bands for id '{sample_id}', "
                        f"got {src.count} at '{img_path}'"
                    )
                img = src.read()  # (C, H, W)
        except RasterioIOError as e:
            raise RuntimeError(
                f"Failed to read image for id '{sample_id}' at '{img_path}': {e}"
            ) from e

        if img.ndim != 3:
            raise ValueError(
                f"Expected image with 3 dimensions (C,H,W) for id '{sample_id}', "
                f"got shape {img.shape}"
            )
        img = img.astype(np.float32, copy=False)

        # --- Percentile Backscatter Clipping (2–98 dB) ---
        img_t = torch.from_numpy(img)

        flat = img_t.view(img_t.shape[0], -1)

        cache_key = {
            "version": _QUANTILE_CACHE_VERSION,
            "dataset_type": "sar",
            "file": _file_cache_fingerprint(img_path),
            "clip_low": 0.02,
            "clip_high": 0.98,
            "band_selection": None,
            "expected_img_bands": self.expected_img_bands,
            "log_transform": self.log_transform,
        }
        cache_path = _quantile_cache_path(
            self.quantile_cache_root, "sar", sample_id, img_path
        )
        cached = _load_quantile_cache(cache_path, cache_key)
        if cached is not None and cached["lower"].numel() == img_t.shape[0]:
            self._record_cache_hit()
            lower_1d = cached["lower"].to(dtype=img_t.dtype, device=img_t.device)
            upper_1d = cached["upper"].to(dtype=img_t.dtype, device=img_t.device)
        else:
            self._record_cache_miss()
            q = torch.tensor([0.02, 0.98], dtype=flat.dtype, device=flat.device)
            quantile_started_at = time.perf_counter() if self._profile_quantiles else None
            quantiles = torch.nanquantile(flat, q, dim=1)
            if quantile_started_at is not None:
                self._record_nanquantile_time(
                    flat, time.perf_counter() - quantile_started_at
                )
            lower_1d = quantiles[0]
            upper_1d = quantiles[1]
            _save_quantile_cache(cache_path, cache_key, lower_1d, upper_1d)

        lower = lower_1d.view(-1, 1, 1)
        upper = upper_1d.view(-1, 1, 1)

        img_t = torch.clamp(img_t, lower, upper)

        img = img_t.numpy()            

        # --- Step 1: flag pixels already invalid after load ---
        invalid = ~np.isfinite(img)  # (C, H, W)

        if self.log_transform:
            # --- Step 2: mask non-positive values before log to prevent -inf/NaN ---
            non_positive = img <= 0
            img = np.where(non_positive, np.nan, img)
            invalid |= non_positive
            img = np.log(img)  # NaN propagates cleanly for masked pixels
            # Catch any new non-finite values produced by the log (e.g. log(0) edge cases)
            invalid |= ~np.isfinite(img)

        # valid_mask: True where ALL bands are valid — shape (H, W)
        valid_mask = ~invalid.any(axis=0)

        # --- Step 3: band-min imputation so tensor stays fully finite ---
        # This is a safety net only; valid_mask ensures these pixels are
        # excluded from training loss and never influence gradients.
        if not valid_mask.all():
            for c in range(img.shape[0]):
                band = img[c]
                finite_vals = band[np.isfinite(band)]
                fill = float(finite_vals.min()) if finite_vals.size > 0 else 0.0
                band[~np.isfinite(band)] = fill
                img[c] = band

        img = self._apply_normalization(img)
        self._record_load_image_time(load_started_at)
        return img, valid_mask

    def _load_mask(self, mask_path: str, expected_hw: Tuple[int, int], sample_id: str) -> np.ndarray:
        try:
            with rasterio.open(mask_path) as src:
                if src.count >= 1:
                    mask = src.read(1)
                else:
                    raise ValueError("Mask has zero bands")
        except RasterioIOError as e:
            raise RuntimeError(f"Failed to read mask for id '{sample_id}' at '{mask_path}': {e}") from e

        if mask.shape != expected_hw:
            raise ValueError(
                f"Mask shape mismatch for id '{sample_id}': "
                f"mask {mask.shape}, image {expected_hw}"
            )

        mask_bin = (mask > 0).astype(np.uint8)
        return mask_bin[None, :, :]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        sample = self.samples[idx]
        sample_id = sample["id"]
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        img_out = self._load_image(img_path, sample_id)
        if isinstance(img_out, tuple) and len(img_out) == 2:
            img, valid_mask = img_out
        else:
            img = img_out
            valid_mask = np.ones((img.shape[1], img.shape[2]), dtype=bool)
        img_tensor = torch.from_numpy(img).float()

        # valid_mask: bool tensor [H, W], False where any band was NaN/Inf.
        # Pass this to the loss function to exclude invalid pixels from gradients.
        valid_mask_tensor = torch.from_numpy(valid_mask)  # dtype=torch.bool

        mask_tensor: Optional[torch.Tensor] = None
        ignore_mask_tensor: Optional[torch.Tensor] = None
        if self.mode != "none":
            if mask_path is None:
                raise RuntimeError(f"Missing mask path for id '{sample_id}'")
            mask_out = self._load_mask(mask_path, (img.shape[1], img.shape[2]), sample_id)
            if isinstance(mask_out, tuple) and len(mask_out) == 2:
                mask, ignore_mask = mask_out
            else:
                mask = mask_out
                ignore_mask = None
            mask_tensor = torch.from_numpy(mask)
            if ignore_mask is not None:
                if not isinstance(ignore_mask, torch.Tensor):
                    ignore_mask_tensor = torch.as_tensor(ignore_mask)
                else:
                    ignore_mask_tensor = ignore_mask
                if ignore_mask_tensor.ndim == 2:
                    ignore_mask_tensor = ignore_mask_tensor.unsqueeze(0)
                elif ignore_mask_tensor.ndim != 3:
                    raise ValueError("ignore_mask must have shape [H,W] or [1,H,W]")
                if ignore_mask_tensor.shape[0] != 1:
                    raise ValueError("ignore_mask must have shape [1,H,W]")
                if tuple(ignore_mask_tensor.shape[-2:]) != tuple(mask_tensor.shape[-2:]):
                    raise ValueError(
                        "ignore_mask spatial shape mismatch: "
                        f"got {tuple(ignore_mask_tensor.shape[-2:])}, "
                        f"expected {tuple(mask_tensor.shape[-2:])}"
                    )
                ignore_mask_tensor = ignore_mask_tensor.to(dtype=torch.bool)

        metadata: Dict[str, Any] = {
            "id": sample_id,
            "img_path": img_path,
            "mask_path": mask_path,
            # Boolean [H, W] pixel-validity mask.  Use this in your loss:
            #   loss = (criterion(pred, target) * valid_mask).sum() / valid_mask.sum().clamp(1)
            "valid_mask": valid_mask_tensor,
        }
        if ignore_mask_tensor is not None:
            metadata["ignore_mask"] = ignore_mask_tensor

        if self.use_time_matched:
            metadata["time_matched_path"] = None
            metadata["time_matched_status"] = None
            tm_path, tm_status = self._time_matched_path(sample)
            if tm_path is None:
                if self.time_matched_missing_policy == "raise":
                    raise RuntimeError(
                        f"Missing time-matched stack for id '{sample_id}' "
                        f"(status={tm_status or 'missing'})"
                    )
                if self.time_matched_missing_policy == "zeros":
                    if not self._warned_missing_tm:
                        warnings.warn("Using zeros for missing time-matched stacks.")
                        self._warned_missing_tm = True
                    tm = np.zeros((8, img.shape[1], img.shape[2]), dtype=np.float32)
                    metadata["time_matched"] = torch.from_numpy(tm)
                metadata["time_matched_status"] = tm_status or "missing"
            else:
                with rasterio.open(tm_path) as src:
                    tm = src.read().astype(np.float32, copy=False)
                if tm.ndim != 3 or tm.shape[0] != 8:
                    raise ValueError(
                        f"Expected time-matched stack with 8 bands for id '{sample_id}', "
                        f"got shape {tm.shape}"
                    )
                metadata["time_matched"] = torch.from_numpy(tm)
                metadata["time_matched_path"] = str(tm_path)
                metadata["time_matched_status"] = tm_status or "ok"

        if self.transforms is not None:
            had_ignore_mask = "ignore_mask" in metadata
            out = self.transforms(img_tensor, mask_tensor, metadata)
            if isinstance(out, tuple) and len(out) == 3:
                img_tensor, mask_tensor, metadata = out
            elif isinstance(out, tuple) and len(out) == 2:
                img_tensor, mask_tensor = out
            else:
                raise ValueError("transforms must return (image, mask, metadata) or (image, mask)")
            if had_ignore_mask and "ignore_mask" not in metadata:
                raise ValueError("dropped metadata['ignore_mask']")

        return img_tensor, mask_tensor, metadata
