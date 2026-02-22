"""
Smoke tests for data-loading utilities:
- recursive ID discovery + duplicate stem warnings
- stats guardrails against transforms/normalization leakage
- expected band count checks
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import SARDataset, compute_running_mean_std, list_ids_from_dir  # noqa: E402

pytestmark = pytest.mark.integration


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


def test_compute_running_mean_std_guardrails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_path = root / "WeaklyLabeled" / "sample_1.tif"
        mask_root = root / "masks"
        mask_path = mask_root / "sample_1.tif"
        _write_tif(img_path, np.ones((2, 4, 4), dtype=np.float32))
        _write_tif(mask_path, np.zeros((4, 4), dtype=np.uint8))

        ds = SARDataset(
            img_root=None,
            mask_root=mask_root,
            ids_or_paths=[img_path],
            mode="strong",
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
        mask_root = root / "masks"
        mask_path = mask_root / "sample_1.tif"
        _write_tif(img_path, np.zeros((2, 4, 4), dtype=np.float32))
        _write_tif(mask_path, np.zeros((4, 4), dtype=np.uint8))

        ds = SARDataset(
            img_root=None,
            mask_root=mask_root,
            ids_or_paths=[img_path],
            mode="strong",
            ids_are_paths=True,
            expected_img_bands=3,
        )

        try:
            _ = ds[0]
            raise AssertionError("Expected ValueError due to mismatched image band count")
        except ValueError as exc:
            assert "Expected 3 image bands" in str(exc)


if __name__ == "__main__":
    test_recursive_discovery_and_duplicates_warning()
    test_compute_running_mean_std_guardrails()
    test_expected_band_checks()
    print("All smoke tests passed.")
