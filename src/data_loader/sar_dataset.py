from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import torch
from torch.utils.data import Dataset

from .patch_dataset import PatchDataset


class Sample(TypedDict):
    id: str
    img_path: str
    mask_path: str


NormalizeCfg = Union[str, Dict[str, Any], None]


def _is_tif_name(name: str) -> bool:
    ext = Path(name).suffix.lower()
    return ext in (".tif", ".tiff")


def _prepare_ids(
    ids_or_paths: Sequence[Union[str, Path]],
    ids_are_paths: bool,
) -> List[Union[str, Path]]:
    if not ids_or_paths:
        raise ValueError("ids_or_paths is empty")

    items: List[Union[str, Path]] = []
    for item in ids_or_paths:
        s = str(item)
        if ids_are_paths:
            p = Path(s)
            if p.suffix.lower() not in (".tif", ".tiff"):
                raise ValueError(
                    "When ids_are_paths=True, items must be .tif/.tiff paths; "
                    f"got '{s}'."
                )
            items.append(p)
        else:
            if _is_tif_name(s):
                raise ValueError(
                    "When ids_are_paths=False, ids must be bare stems (no extension); "
                    f"got '{s}'."
                )
            if Path(s).name != s:
                raise ValueError(
                    "When ids_are_paths=False, ids must be bare stems (no path separators); "
                    f"got '{s}'."
                )
            items.append(s)
    return items


def _require_dir(path: Optional[Union[str, Path]], label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required")
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"{label} does not exist or is not a directory: {root}")
    return root


class SARDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 SAR tiles with hand-labeled flood masks.

    Assumptions:
    - ids_or_paths are sample IDs (stems) when ids_are_paths=False.
    - Image path resolves to img_root/<id>.tif or img_root/<id>.tiff.
    - Masks are always present under mask_root with the same ID (or a suffix map).
    - mode "strong" is the default; mode "weak" is retained for compatibility and
      uses the same mask loading behavior.
    """

    def __init__(
        self,
        img_root: Optional[Union[str, Path]],
        mask_root: Optional[Union[str, Path]],
        ids_or_paths: Sequence[Union[str, Path]],
        mode: str = "strong",
        transforms: Optional[Callable[..., Any]] = None,
        normalize_cfg: NormalizeCfg = None,
        log_transform: bool = False,
        validate: bool = False,
        ids_are_paths: bool = False,
        mask_id_suffix_map: Optional[Dict[str, str]] = None,
        expected_img_bands: Optional[int] = None,
    ) -> None:
        if mode not in {"strong", "weak"}:
            raise ValueError(f"mode must be 'strong' or 'weak'; got {mode}")

        self.mode = mode
        self.transforms = transforms
        self.log_transform = bool(log_transform)
        self.ids_are_paths = bool(ids_are_paths)
        self.mask_id_suffix_map = dict(mask_id_suffix_map) if mask_id_suffix_map else None

        if self.ids_are_paths and img_root is not None:
            raise ValueError("img_root must be None when ids_are_paths=True")

        self.img_root = None if self.ids_are_paths else _require_dir(img_root, "img_root")
        self.mask_root = _require_dir(mask_root, "mask_root")

        self.expected_img_bands = expected_img_bands
        if self.expected_img_bands is not None and self.expected_img_bands <= 0:
            raise ValueError("expected_img_bands must be a positive integer")

        self._normalize_type: str = "none"
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._parse_normalize_cfg(normalize_cfg)

        prepared = _prepare_ids(ids_or_paths, self.ids_are_paths)
        self.samples: List[Sample] = self._build_samples(prepared)
        if validate:
            self._validate_samples(self.samples)

    def with_patches(
        self,
        patch_size: int = 256,
        overlap: float = 0.2,
        skip_mostly_nodata: bool = False,
        nodata_threshold: float = 0.0,
    ) -> PatchDataset:
        return PatchDataset(
            self,
            patch_size=patch_size,
            overlap=overlap,
            skip_mostly_nodata=skip_mostly_nodata,
            nodata_threshold=nodata_threshold,
        )

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
            raise ValueError(
                f"normalize_cfg type must be 'none' or 'zscore', got {self._normalize_type}"
            )

    def _resolve_raster_path(self, root: Path, name_or_id: str) -> Path:
        tif = root / f"{name_or_id}.tif"
        tiff = root / f"{name_or_id}.tiff"
        if tif.exists():
            return tif
        if tiff.exists():
            return tiff
        raise FileNotFoundError(f"Missing raster for '{name_or_id}' under {root}")

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
            if self.ids_are_paths:
                img_path = Path(item)
                sample_id = img_path.stem
            else:
                sample_id = str(item)
                if self.img_root is None:
                    raise ValueError("img_root is required when ids_are_paths=False")
                img_path = self._resolve_raster_path(self.img_root, sample_id)

            mask_id = self._resolve_mask_id(sample_id)
            mask_path = self._resolve_raster_path(self.mask_root, mask_id)

            samples.append(
                {
                    "id": sample_id,
                    "img_path": str(img_path),
                    "mask_path": str(mask_path),
                }
            )
        return samples

    def _validate_samples(self, samples: List[Sample]) -> None:
        for sample in samples:
            img_path = sample["img_path"]
            mask_path = sample["mask_path"]
            sample_id = sample["id"]
            with rasterio.open(img_path) as src:
                img_h, img_w = src.height, src.width
                if src.count < 1:
                    raise ValueError(f"Image has zero bands (id='{sample_id}')")
            with rasterio.open(mask_path) as msrc:
                mask_h, mask_w = msrc.height, msrc.width
            if (img_h, img_w) != (mask_h, mask_w):
                raise ValueError(
                    f"Mask shape mismatch for id '{sample_id}': "
                    f"mask {(mask_h, mask_w)} vs image {(img_h, img_w)}"
                )

    def _apply_normalization(self, img: np.ndarray) -> np.ndarray:
        if self._normalize_type == "none":
            return img

        if self._normalize_type == "zscore":
            if self._norm_mean is None or self._norm_std is None:
                raise ValueError("normalize_cfg for 'zscore' must include 'mean' and 'std'")
            c = img.shape[0]
            if self._norm_mean.size != c or self._norm_std.size != c:
                raise ValueError(
                    "Normalization stats length mismatch: got mean/std length "
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
        except RasterioIOError as exc:
            raise RuntimeError(
                f"Failed to read image for id '{sample_id}' at '{img_path}': {exc}"
            ) from exc

        if img.ndim != 3:
            raise ValueError(
                f"Expected image with 3 dimensions (C,H,W) for id '{sample_id}', "
                f"got shape {img.shape}"
            )
        img = img.astype(np.float32, copy=False)

        if self.log_transform:
            img = np.log1p(np.abs(img))

        img = self._apply_normalization(img)
        if not np.isfinite(img).all():
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        return img

    def _load_mask(
        self,
        mask_path: str,
        expected_hw: Tuple[int, int],
        sample_id: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            with rasterio.open(mask_path) as src:
                if src.count >= 1:
                    mask = src.read(1)
                else:
                    raise ValueError("Mask has zero bands")
        except RasterioIOError as exc:
            raise RuntimeError(
                f"Failed to read mask for id '{sample_id}' at '{mask_path}': {exc}"
            ) from exc

        if mask.shape != expected_hw:
            raise ValueError(
                f"Mask shape mismatch for id '{sample_id}': "
                f"mask {mask.shape}, image {expected_hw}"
            )

        ignore = mask < 0
        mask_bin = (mask > 0).astype(np.uint8)
        ignore_mask = ignore.astype(bool, copy=False)[None, :, :]
        return mask_bin[None, :, :], ignore_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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

        mask, ignore_mask = self._load_mask(mask_path, (img.shape[1], img.shape[2]), sample_id)
        mask_tensor = torch.from_numpy(mask)
        metadata["ignore_mask"] = torch.from_numpy(ignore_mask)

        if self.transforms is not None:
            had_ignore = "ignore_mask" in metadata
            out = self.transforms(img_tensor, mask_tensor, metadata)
            if isinstance(out, tuple) and len(out) == 3:
                img_tensor, mask_tensor, metadata = out
            elif isinstance(out, tuple) and len(out) == 2:
                img_tensor, mask_tensor = out
            else:
                raise ValueError("transforms must return (image, mask, metadata) or (image, mask)")
            if had_ignore and "ignore_mask" not in metadata:
                raise ValueError(
                    "transforms dropped metadata['ignore_mask']; "
                    "return updated metadata or do not mutate it "
                    f"(id='{sample_id}')."
                )
            if "ignore_mask" in metadata:
                ignore_mask_t = metadata["ignore_mask"]
                if not isinstance(ignore_mask_t, torch.Tensor):
                    ignore_mask_t = torch.as_tensor(ignore_mask_t)
                if ignore_mask_t.ndim == 2:
                    ignore_mask_t = ignore_mask_t.unsqueeze(0)
                if ignore_mask_t.ndim != 3 or ignore_mask_t.shape[0] != 1:
                    raise ValueError(
                        f"ignore_mask must have shape [1,H,W]; got {tuple(ignore_mask_t.shape)} "
                        f"(id='{sample_id}')"
                    )
                if mask_tensor.ndim == 2:
                    expected_hw = tuple(mask_tensor.shape)
                elif mask_tensor.ndim == 3:
                    expected_hw = tuple(mask_tensor.shape[1:])
                else:
                    raise ValueError(
                        f"mask must have shape [1,H,W] or [H,W]; got {tuple(mask_tensor.shape)} "
                        f"(id='{sample_id}')"
                    )
                if tuple(ignore_mask_t.shape[1:]) != expected_hw:
                    raise ValueError(
                        "ignore_mask spatial shape mismatch: "
                        f"got {tuple(ignore_mask_t.shape[1:])}, expected {expected_hw} "
                        f"(id='{sample_id}')"
                    )
                metadata["ignore_mask"] = ignore_mask_t

        return img_tensor, mask_tensor, metadata
