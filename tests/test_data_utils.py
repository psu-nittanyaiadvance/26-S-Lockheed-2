"""
Smoke tests for data-loading utilities:
- recursive ID discovery + duplicate stem warnings
- time-matched zeros warning constant + skip policy filtering
- stats guardrails against transforms/normalization leakage
- expected band count checks
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import (  # noqa: E402
    SARDataset,
    TIME_MATCHED_ZEROS_WARNING,
    compute_running_mean_std,
    list_ids_from_dir,
    validate_time_matched,
)


def _write_tif(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[None, :, :]
    count, height, width = data.shape
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        transform=transform,
    ) as dst:
        dst.write(data)


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sample_id,path,status\n", encoding="utf-8")


def test_recursive_discovery_and_duplicates_warning() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_tif(root / "nested" / "tile_a.tif", np.zeros((2, 4, 4), dtype=np.float32))
        _write_tif(root / "dup1" / "tile_dup.tif", np.zeros((2, 4, 4), dtype=np.float32))
        _write_tif(root / "dup2" / "tile_dup.tif", np.zeros((2, 4, 4), dtype=np.float32))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ids = list_ids_from_dir(root, recursive=True)

        assert "tile_a" in ids
        assert "tile_dup" in ids
        dup_warns = [str(w.message) for w in caught if "Duplicate stem" in str(w.message)]
        assert len(dup_warns) == 1


def test_time_matched_zeros_warning_and_validator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_path = root / "WeaklyLabeled" / "sample_1.tif"
        _write_tif(img_path, np.zeros((2, 4, 4), dtype=np.float32))

        manifest_path = root / "tm_manifest.csv"
        _write_manifest(manifest_path)

        ds = SARDataset(
            img_root=None,
            mask_root=None,
            ids_or_paths=[img_path],
            mode="none",
            ids_are_paths=True,
            use_time_matched=True,
            time_matched_root=root / "tm",
            time_matched_manifest=manifest_path,
            time_matched_missing_policy="zeros",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = ds[0]
            _ = ds[0]

        warn_msgs = [str(w.message) for w in caught]
        assert warn_msgs.count(TIME_MATCHED_ZEROS_WARNING) == 1

        validate_time_matched(ds, n=1)


def test_time_matched_skip_filters_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_ok = root / "WeaklyLabeled" / "ok_1.tif"
        img_missing = root / "WeaklyLabeled" / "missing_1.tif"
        _write_tif(img_ok, np.zeros((2, 4, 4), dtype=np.float32))
        _write_tif(img_missing, np.zeros((2, 4, 4), dtype=np.float32))

        tm_root = root / "tm"
        _write_tif(
            tm_root / "WeaklyLabeled" / "ok_1.tif",
            np.zeros((8, 4, 4), dtype=np.float32),
        )

        manifest_path = root / "tm_manifest.csv"
        _write_manifest(manifest_path)

        ds = SARDataset(
            img_root=None,
            mask_root=None,
            ids_or_paths=[img_ok, img_missing],
            mode="none",
            ids_are_paths=True,
            use_time_matched=True,
            time_matched_root=tm_root,
            time_matched_manifest=manifest_path,
            time_matched_missing_policy="skip",
        )

        assert len(ds) == 1
        _, _, meta = ds[0]
        assert meta["id"] == "ok_1"


def test_compute_running_mean_std_guardrails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_path = root / "WeaklyLabeled" / "sample_1.tif"
        _write_tif(img_path, np.ones((2, 4, 4), dtype=np.float32))

        ds = SARDataset(
            img_root=None,
            mask_root=None,
            ids_or_paths=[img_path],
            mode="none",
            ids_are_paths=True,
            transforms=lambda i, m, meta: (i, m, meta),
        )

        try:
            compute_running_mean_std(ds, require_no_transforms=True)
            raise AssertionError("Expected ValueError due to transforms being enabled")
        except ValueError as exc:
            assert "Transforms are enabled" in str(exc)


def test_expected_band_checks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_path = root / "WeaklyLabeled" / "sample_1.tif"
        _write_tif(img_path, np.zeros((2, 4, 4), dtype=np.float32))

        ds = SARDataset(
            img_root=None,
            mask_root=None,
            ids_or_paths=[img_path],
            mode="none",
            ids_are_paths=True,
            expected_img_bands=3,
        )

        try:
            _ = ds[0]
            raise AssertionError("Expected ValueError due to mismatched image band count")
        except ValueError as exc:
            assert "Expected 3 image bands" in str(exc)

        tm_root = root / "tm"
        _write_tif(
            tm_root / "WeaklyLabeled" / "sample_1.tif",
            np.zeros((7, 4, 4), dtype=np.float32),
        )
        manifest_path = root / "tm_manifest.csv"
        _write_manifest(manifest_path)

        ds_tm = SARDataset(
            img_root=None,
            mask_root=None,
            ids_or_paths=[img_path],
            mode="none",
            ids_are_paths=True,
            use_time_matched=True,
            time_matched_root=tm_root,
            time_matched_manifest=manifest_path,
            expected_time_matched_bands=8,
        )

        try:
            _ = ds_tm[0]
            raise AssertionError("Expected ValueError due to mismatched time-matched band count")
        except ValueError as exc:
            assert "Expected time-matched stack with 8 bands" in str(exc)


if __name__ == "__main__":
    test_recursive_discovery_and_duplicates_warning()
    test_time_matched_zeros_warning_and_validator()
    test_time_matched_skip_filters_missing()
    test_compute_running_mean_std_guardrails()
    test_expected_band_checks()
    print("All smoke tests passed.")
