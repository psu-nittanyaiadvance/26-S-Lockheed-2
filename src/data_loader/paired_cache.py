from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .combined_manifest import load_combined_manifest_samples


PAIRED_CACHE_FORMAT = "paired_preprocessed_tensor_cache_v1"
PAIRED_CACHE_INDEX = "index.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_identity(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    identity: Dict[str, Any] = {"path": str(p)}
    try:
        stat = p.stat()
    except FileNotFoundError:
        identity["exists"] = False
        return identity
    identity.update(
        {
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
    return identity


def paired_preprocessing_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_paired_preprocessing_config(
    combined_root: str | Path,
    *,
    sar_dataset_kwargs: Optional[Mapping[str, Any]] = None,
    optical_dataset_kwargs: Optional[Mapping[str, Any]] = None,
    require_spatial_match: bool = True,
    validate_manifest: bool = False,
) -> Dict[str, Any]:
    """
    Build the deterministic-preprocessing fingerprint inputs for paired pretraining.

    The config intentionally includes manifest row order plus raw asset file
    identity, because changing either changes the cached training contract even
    when the command-line preprocessing flags stay the same.
    """
    samples = load_combined_manifest_samples(combined_root, require_label=False)
    manifest_path = samples[0].manifest_path

    config: Dict[str, Any] = {
        "format": PAIRED_CACHE_FORMAT,
        "combined_root": str(combined_root),
        "manifest": _file_identity(manifest_path),
        "require_spatial_match": bool(require_spatial_match),
        "validate_manifest": bool(validate_manifest),
        "return_mode": "paired",
        "sar_dataset": {
            "class": "SARDataset",
            "mode": "none",
            "kwargs": _jsonable(dict(sar_dataset_kwargs or {})),
        },
        "optical_dataset": {
            "class": "OpticalDataset",
            "mode": "none",
            "kwargs": _jsonable(dict(optical_dataset_kwargs or {})),
        },
        "samples": [
            {
                "sample_id": sample.sample_id,
                "manifest_index": sample.manifest_index,
                "sar": _file_identity(sample.sar_path),
                "optical": _file_identity(sample.optical_path),
            }
            for sample in samples
        ],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    config["preprocessing_config_digest"] = paired_preprocessing_digest(config)
    return config


def _load_pt(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _as_bool_hw(mask: Any, *, sample_id: str, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        raise ValueError(f"{name} for id='{sample_id}' must have shape [H,W], got {tuple(mask.shape)}")
    return mask.to(dtype=torch.bool).cpu()


def _training_safe_meta(
    pair: Mapping[str, Any],
    *,
    preprocessing_config_digest: str,
) -> Dict[str, Any]:
    metadata = dict(pair["metadata"])
    metadata.pop("valid_mask", None)
    metadata.pop("sar_valid_mask", None)
    metadata.pop("optical_valid_mask", None)
    metadata.pop("mask_path", None)
    metadata.pop("label_path", None)

    sample_id = str(metadata.get("paired_sample_id", metadata.get("id", "")))
    metadata.update(
        {
            "id": sample_id,
            "paired_sample_id": sample_id,
            "sar_img_path": metadata.get("sar_img_path"),
            "optical_img_path": metadata.get("optical_img_path"),
            "manifest_path": metadata.get("manifest_path"),
            "manifest_row_index": metadata.get("manifest_row_index"),
            "n_sar_bands": int(pair["sar_image"].shape[0]),
            "n_optical_bands": int(pair["optical_image"].shape[0]),
            "preprocessing_config_digest": preprocessing_config_digest,
            "modality": "paired_multimodal_cached",
        }
    )
    return metadata


def _sample_from_live_pair(
    pair: Mapping[str, Any],
    *,
    preprocessing_config_digest: str,
) -> Dict[str, Any]:
    metadata = _training_safe_meta(pair, preprocessing_config_digest=preprocessing_config_digest)
    sample_id = str(metadata["paired_sample_id"])
    sar = pair["sar_image"].detach().cpu().to(dtype=torch.float32).contiguous()
    optical = pair["optical_image"].detach().cpu().to(dtype=torch.float32).contiguous()
    sar_valid = _as_bool_hw(pair["metadata"]["sar_valid_mask"], sample_id=sample_id, name="sar_valid_mask")
    optical_valid = _as_bool_hw(
        pair["metadata"]["optical_valid_mask"],
        sample_id=sample_id,
        name="optical_valid_mask",
    )

    return {
        "sar": sar,
        "optical": optical,
        "valid_mask": (sar_valid & optical_valid).contiguous(),
        "sar_valid_mask": sar_valid.contiguous(),
        "optical_valid_mask": optical_valid.contiguous(),
        "meta": metadata,
    }


def build_paired_preprocessed_cache(
    source_dataset: Dataset,
    cache_dir: str | Path,
    *,
    preprocessing_config: Mapping[str, Any],
    max_samples: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Materialize pre-augmentation paired tensors from the live raster dataset.

    ``source_dataset`` is expected to be the strict ``FusedDataset`` paired path.
    The function calls ``get_paired_item(..., require_no_transforms=True)`` so
    cache generation inherits the existing SAR and optical preprocessing code.
    """
    if not hasattr(source_dataset, "get_paired_item"):
        raise TypeError("source_dataset must expose get_paired_item()")

    cache_path = Path(cache_dir)
    index_path = cache_path / PAIRED_CACHE_INDEX
    samples_dir = cache_path / "samples"
    if index_path.exists() and not overwrite:
        raise FileExistsError(
            f"Paired cache index already exists at {index_path}; pass overwrite=True to rebuild"
        )

    cache_path.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    digest = str(preprocessing_config.get("preprocessing_config_digest") or paired_preprocessing_digest(preprocessing_config))
    total = len(source_dataset)  # type: ignore[arg-type]
    limit = total if max_samples is None else min(int(max_samples), total)
    if limit < 0:
        raise ValueError("max_samples must be non-negative")

    records: List[Dict[str, Any]] = []
    for idx in range(limit):
        pair = source_dataset.get_paired_item(idx, require_no_transforms=True)  # type: ignore[attr-defined]
        sample = _sample_from_live_pair(pair, preprocessing_config_digest=digest)
        record_relpath = Path("samples") / f"{idx:08d}.pt"
        torch.save(sample, str(cache_path / record_relpath))

        meta = sample["meta"]
        records.append(
            {
                "index": idx,
                "id": meta["id"],
                "paired_sample_id": meta["paired_sample_id"],
                "manifest_row_index": meta.get("manifest_row_index"),
                "sar_img_path": meta.get("sar_img_path"),
                "optical_img_path": meta.get("optical_img_path"),
                "path": str(record_relpath).replace("\\", "/"),
            }
        )

    index = {
        "format": PAIRED_CACHE_FORMAT,
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "num_samples": limit,
        "source_num_samples": total,
        "preprocessing_config": _jsonable(dict(preprocessing_config)),
        "preprocessing_config_digest": digest,
        "records": records,
    }
    tmp_index = index_path.with_suffix(".json.tmp")
    tmp_index.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    tmp_index.replace(index_path)
    return index


class CachedPairedDataset(Dataset):
    """Dataset for preprocessed paired SAR/optical tensor cache records."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_preprocessing_config_digest: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / PAIRED_CACHE_INDEX
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Paired cache index not found: {self.index_path}")

        self.index: Dict[str, Any] = json.loads(self.index_path.read_text(encoding="utf-8"))
        if self.index.get("format") != PAIRED_CACHE_FORMAT:
            raise ValueError(
                f"Unsupported paired cache format {self.index.get('format')!r}; "
                f"expected {PAIRED_CACHE_FORMAT!r}"
            )
        self.preprocessing_config_digest = str(self.index.get("preprocessing_config_digest", ""))
        if (
            expected_preprocessing_config_digest is not None
            and self.preprocessing_config_digest != expected_preprocessing_config_digest
        ):
            raise ValueError(
                "Paired cache preprocessing digest mismatch: "
                f"cache={self.preprocessing_config_digest} "
                f"expected={expected_preprocessing_config_digest}"
            )

        self.records: List[Dict[str, Any]] = list(self.index.get("records", []))
        if int(self.index.get("num_samples", len(self.records))) != len(self.records):
            raise ValueError("Paired cache index num_samples does not match records length")

    def __len__(self) -> int:
        return len(self.records)

    def load_record(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        path = self.cache_dir / str(record["path"])
        sample = _load_pt(path)
        required = {"sar", "optical", "valid_mask", "sar_valid_mask", "optical_valid_mask", "meta"}
        missing = sorted(required - set(sample))
        if missing:
            raise KeyError(f"Cached paired sample {path} is missing keys {missing}")
        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.load_record(idx)
        meta = dict(sample["meta"])
        meta["preprocessing_config_digest"] = self.preprocessing_config_digest
        return {
            "sar": sample["sar"].to(dtype=torch.float32),
            "optical": sample["optical"].to(dtype=torch.float32),
            "valid_mask": sample["valid_mask"].to(dtype=torch.bool),
            "meta": meta,
        }

    def __repr__(self) -> str:
        return (
            f"CachedPairedDataset(n={len(self)}, "
            f"digest={self.preprocessing_config_digest[:12]!r}, "
            f"cache_dir={str(self.cache_dir)!r})"
        )


def validate_paired_cache_parity(
    live_dataset: Dataset,
    cached_dataset: CachedPairedDataset,
    *,
    indices: Optional[Iterable[int]] = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Dict[str, Any]:
    if not hasattr(live_dataset, "get_paired_item"):
        raise TypeError("live_dataset must expose get_paired_item()")

    if indices is None:
        check_indices: Sequence[int] = range(min(len(live_dataset), len(cached_dataset)))  # type: ignore[arg-type]
    else:
        check_indices = list(indices)

    mismatches: List[str] = []
    meta_keys = [
        "id",
        "paired_sample_id",
        "sar_img_path",
        "optical_img_path",
        "manifest_path",
        "manifest_row_index",
        "n_sar_bands",
        "n_optical_bands",
    ]

    for idx in check_indices:
        live_pair = live_dataset.get_paired_item(idx, require_no_transforms=True)  # type: ignore[attr-defined]
        live_sample = _sample_from_live_pair(
            live_pair,
            preprocessing_config_digest=cached_dataset.preprocessing_config_digest,
        )
        cached_record = cached_dataset.load_record(idx)
        cached_sample = cached_dataset[idx]

        for key in ("sar", "optical"):
            if not torch.allclose(live_sample[key], cached_sample[key], atol=atol, rtol=rtol):
                mismatches.append(f"idx={idx} tensor {key} differs")
        for key in ("valid_mask", "sar_valid_mask", "optical_valid_mask"):
            cached_value = cached_sample[key] if key == "valid_mask" else cached_record[key]
            if not torch.equal(live_sample[key].to(dtype=torch.bool), cached_value.to(dtype=torch.bool)):
                mismatches.append(f"idx={idx} mask {key} differs")

        for key in meta_keys:
            if live_sample["meta"].get(key) != cached_sample["meta"].get(key):
                mismatches.append(
                    f"idx={idx} meta {key} differs: "
                    f"live={live_sample['meta'].get(key)!r} cached={cached_sample['meta'].get(key)!r}"
                )

    return {
        "checked": len(check_indices),
        "ok": not mismatches,
        "mismatches": mismatches,
        "atol": atol,
        "rtol": rtol,
    }
