from __future__ import annotations

import csv
import os
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.errors import RasterioIOError


class Sample(TypedDict):
    id: str
    img_path: str
    mask_path: Optional[str]


NormalizeCfg = Union[str, Dict[str, Any], None]


TIME_MATCHED_ZEROS_WARNING = "Using zeros for missing time-matched stacks."
DEFAULT_SPLIT_TOKENS = (
    "WeaklyLabeled",
    "HandLabeled",
    "weak",
    "strong",
    "Weak",
    "Strong",
)


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


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


def _infer_split_from_path(path: str, split_tokens: Sequence[str]) -> str:
    parts = Path(path).parts
    norm_parts = [_normalize_token(p) for p in parts]
    for token in split_tokens:
        norm_token = _normalize_token(token)
        if norm_token and norm_token in norm_parts:
            return token
    for token in split_tokens:
        norm_token = _normalize_token(token)
        if not norm_token:
            continue
        if any(norm_token in part for part in norm_parts):
            return token
    return "all"


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
    - mask_id_suffix_map can rewrite the image stem when resolving mask filenames.

    Time-matched behavior (use_time_matched=True):
    - Reads a stack with expected_time_matched_bands (default 8) from
      time_matched_root/{split}/{id}.tif or uses the time_matched_manifest if present.
    - Split is inferred from the image path using split_tokens; metadata includes split_inferred.
    - Metadata always includes time_matched_status and time_matched_path (even if None).

    Missing policies for time-matched stacks:
    - "zeros": return an all-zero 8-band stack (warning emitted once).
    - "skip": drop samples without a valid time-matched stack at init time.
    - "raise": raise at __getitem__ if a time-matched stack is missing.

    The Dataset does not attempt to map weak<->strong IDs; it only uses the
    provided IDs or image paths.

    Optional band-count checks:
    - expected_img_bands validates the number of SAR bands at read time.
    - expected_time_matched_bands validates the time-matched stack band count.
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
        mask_id_suffix_map: Optional[Dict[str, str]] = None,
        use_time_matched: bool = False,
        time_matched_root: Union[str, Path] = "data/derived/gee_time_matched",
        time_matched_manifest: Union[str, Path] = "data/derived/gee_time_matched_manifest.csv",
        time_matched_missing_policy: str = "zeros",
        split_tokens: Optional[Sequence[str]] = DEFAULT_SPLIT_TOKENS,
        expected_img_bands: Optional[int] = None,
        expected_time_matched_bands: int = 8,
    ) -> None:
        if mode not in {"weak", "strong", "none"}:
            raise ValueError(f"mode must be one of 'weak', 'strong', 'none'; got {mode}")

        if not ids_or_paths:
            raise ValueError("ids_or_paths is empty")

        self.mode = mode
        self.transforms = transforms
        self.log_transform = bool(log_transform)
        self.mask_id_suffix_map = dict(mask_id_suffix_map) if mask_id_suffix_map else None

        if ids_are_paths is None:
            ids_are_paths = _infer_ids_are_paths(ids_or_paths)
        self.ids_are_paths = bool(ids_are_paths)

        self.img_root = Path(img_root) if img_root is not None else None
        self.mask_root = Path(mask_root) if mask_root is not None else None

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
        if split_tokens is None:
            split_tokens = DEFAULT_SPLIT_TOKENS
        self.split_tokens = tuple(split_tokens)
        self.expected_img_bands = expected_img_bands
        self.expected_time_matched_bands = int(expected_time_matched_bands)
        if self.time_matched_missing_policy not in {"zeros", "skip", "raise"}:
            raise ValueError(
                "time_matched_missing_policy must be one of {'zeros','skip','raise'}"
            )
        if self.expected_img_bands is not None and self.expected_img_bands <= 0:
            raise ValueError("expected_img_bands must be a positive integer")
        if self.expected_time_matched_bands <= 0:
            raise ValueError("expected_time_matched_bands must be a positive integer")
        if not self.split_tokens:
            raise ValueError("split_tokens must contain at least one token")
        self._tm_index: Dict[str, Dict[str, str]] = (
            self._load_time_matched_index() if self.use_time_matched else {}
        )
        self._warned_missing_tm = False

        self._normalize_type: str = "none"
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._parse_normalize_cfg(normalize_cfg)

        self.samples: List[Sample] = self._build_samples(ids_or_paths)
        if self.use_time_matched:
            split_all = sum(
                1
                for s in self.samples
                if _infer_split_from_path(s["img_path"], self.split_tokens) == "all"
            )
            if split_all:
                ratio = split_all / max(len(self.samples), 1)
                if ratio >= 0.5:
                    warnings.warn(
                        "Split inference fell back to 'all' for "
                        f"{split_all}/{len(self.samples)} samples. "
                        "Consider passing split_tokens that match your directory structure."
                    )
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

    def _time_matched_path(
        self, sample: Sample, split_override: Optional[str] = None
    ) -> Tuple[Optional[Path], Optional[str]]:
        sample_id = sample["id"]
        row = self._tm_index.get(sample_id)
        if row and row.get("status") == "ok" and row.get("path"):
            path = Path(row["path"])
            if path.exists():
                return path, "ok"
        split = split_override or _infer_split_from_path(sample["img_path"], self.split_tokens)
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

    def _load_image(self, img_path: str, sample_id: str) -> np.ndarray:
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

        if self.log_transform:
            img = np.log1p(np.abs(img))

        img = self._apply_normalization(img)
        return img

    def _load_mask(
        self,
        mask_path: str,
        expected_hw: Tuple[int, int],
        sample_id: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

        ignore = mask < 0
        mask_bin = (mask > 0).astype(np.uint8)
        ignore_mask = ignore.astype(bool, copy=False)
        if not ignore_mask.any():
            ignore_mask = None
        else:
            ignore_mask = ignore_mask[None, :, :]
        return mask_bin[None, :, :], ignore_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        sample = self.samples[idx]
        sample_id = sample["id"]
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        metadata: Dict[str, Any] = {
            "id": sample_id,
            "img_path": img_path,
            "mask_path": mask_path,
        }

        img = self._load_image(img_path, sample_id)
        img_tensor = torch.from_numpy(img).float()

        mask_tensor: Optional[torch.Tensor] = None
        if self.mode != "none":
            if mask_path is None:
                raise RuntimeError(f"Missing mask path for id '{sample_id}'")
            mask, ignore_mask = self._load_mask(mask_path, (img.shape[1], img.shape[2]), sample_id)
            mask_tensor = torch.from_numpy(mask)
            if ignore_mask is not None:
                metadata["ignore_mask"] = torch.from_numpy(ignore_mask)

        if self.use_time_matched:
            metadata["time_matched_path"] = None
            metadata["time_matched_status"] = None
            split_inferred = _infer_split_from_path(img_path, self.split_tokens)
            metadata["split_inferred"] = split_inferred
            tm_path, tm_status = self._time_matched_path(sample, split_override=split_inferred)
            if tm_path is None:
                if self.time_matched_missing_policy == "raise":
                    raise RuntimeError(
                        f"Missing time-matched stack for id '{sample_id}' "
                        f"(status={tm_status or 'missing'})"
                    )
                if self.time_matched_missing_policy == "zeros":
                    if not self._warned_missing_tm:
                        warnings.warn(TIME_MATCHED_ZEROS_WARNING)
                        self._warned_missing_tm = True
                    tm = np.zeros(
                        (self.expected_time_matched_bands, img.shape[1], img.shape[2]),
                        dtype=np.float32,
                    )
                    metadata["time_matched"] = torch.from_numpy(tm)
                metadata["time_matched_status"] = tm_status or "missing"
            else:
                with rasterio.open(tm_path) as src:
                    tm = src.read().astype(np.float32, copy=False)
                if tm.ndim != 3 or tm.shape[0] != self.expected_time_matched_bands:
                    raise ValueError(
                        f"Expected time-matched stack with {self.expected_time_matched_bands} "
                        f"bands for id '{sample_id}', got shape {tm.shape}"
                    )
                metadata["time_matched"] = torch.from_numpy(tm)
                metadata["time_matched_path"] = str(tm_path)
                metadata["time_matched_status"] = tm_status or "ok"

        if self.transforms is not None:
            out = self.transforms(img_tensor, mask_tensor, metadata)
            if isinstance(out, tuple) and len(out) == 3:
                img_tensor, mask_tensor, metadata = out
            elif isinstance(out, tuple) and len(out) == 2:
                img_tensor, mask_tensor = out
            else:
                raise ValueError("transforms must return (image, mask, metadata) or (image, mask)")

        return img_tensor, mask_tensor, metadata
