from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import OpticalDataset  # noqa: E402

pytestmark = pytest.mark.integration


def _write_tif(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[None, :, :]
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)


def test_requires_mask_root_for_labeled_modes(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    img_root.mkdir()

    with pytest.raises(ValueError, match="mask_root is required"):
        OpticalDataset(
            img_root=img_root,
            mask_root=None,
            ids_or_paths=["tile_001"],
            mode="strong",
        )


def test_builds_samples_from_ids(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    mask_root = tmp_path / "mask"
    img_root.mkdir()
    mask_root.mkdir()

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=mask_root,
        ids_or_paths=["tile_001", "tile_002"],
        mode="strong",
    )

    assert [sample["id"] for sample in ds.samples] == ["tile_001", "tile_002"]
    assert Path(ds.samples[0]["img_path"]).name == "tile_001.tif"
    assert Path(ds.samples[0]["mask_path"]).name == "tile_001.tif"


def test_validate_drops_bad_shape_mismatch_samples(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    mask_root = tmp_path / "mask"
    img_root.mkdir()
    mask_root.mkdir()

    _write_tif(img_root / "good.tif", np.ones((2, 4, 4), dtype=np.uint16))
    _write_tif(mask_root / "good.tif", np.ones((4, 4), dtype=np.uint8))
    _write_tif(img_root / "bad.tif", np.ones((2, 4, 4), dtype=np.uint16))
    _write_tif(mask_root / "bad.tif", np.ones((3, 4), dtype=np.uint8))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = OpticalDataset(
            img_root=img_root,
            mask_root=mask_root,
            ids_or_paths=["good", "bad"],
            mode="strong",
            validate=True,
        )

    assert len(ds) == 1
    assert ds.samples[0]["id"] == "good"
    assert any("Dropping sample 'bad'" in str(w.message) for w in caught)


def test_load_image_scales_dn_to_reflectance(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    img_root.mkdir()
    _write_tif(img_root / "tile.tif", np.full((2, 2, 2), 5000, dtype=np.uint16))

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=None,
        ids_or_paths=["tile"],
        mode="none",
        reflectance_clip_percentile=0.0,
    )

    img, valid_mask = ds._load_image(str(img_root / "tile.tif"), "tile")

    assert img.dtype == np.float32
    assert np.allclose(img, 0.5)
    assert valid_mask.dtype == np.bool_
    assert valid_mask.all()


def test_load_image_flags_all_zero_pixels_invalid(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    img_root.mkdir()
    data = np.full((2, 2, 2), 2500, dtype=np.uint16)
    data[:, 0, 0] = 0
    _write_tif(img_root / "tile.tif", data)

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=None,
        ids_or_paths=["tile"],
        mode="none",
        reflectance_clip_percentile=0.0,
    )

    img, valid_mask = ds._load_image(str(img_root / "tile.tif"), "tile")

    assert valid_mask.shape == (2, 2)
    assert not valid_mask[0, 0]
    assert valid_mask[0, 1]
    assert np.isfinite(img).all()


def test_cloud_mask_marks_invalid_pixels(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    scl_root = tmp_path / "scl"
    img_root.mkdir()
    scl_root.mkdir()
    _write_tif(img_root / "tile.tif", np.full((2, 2, 2), 2500, dtype=np.uint16))
    scl = np.zeros((2, 2), dtype=np.uint8)
    scl[1, 0] = 3
    _write_tif(scl_root / "tile.tif", scl)

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=None,
        ids_or_paths=["tile"],
        mode="none",
        cloud_mask_root=scl_root,
        reflectance_clip_percentile=0.0,
    )

    _img, valid_mask = ds._load_image(str(img_root / "tile.tif"), "tile")

    assert not valid_mask[1, 0]
    assert valid_mask[0, 0]


def test_expected_img_bands_is_enforced(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    img_root.mkdir()
    _write_tif(img_root / "tile.tif", np.ones((2, 2, 2), dtype=np.uint16))

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=None,
        ids_or_paths=["tile"],
        mode="none",
        expected_img_bands=3,
    )

    with pytest.raises(ValueError, match="Expected 3 bands"):
        _ = ds[0]


def test_s2_band_selection_out_of_range_raises(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    img_root.mkdir()
    _write_tif(img_root / "tile.tif", np.ones((2, 2, 2), dtype=np.uint16))

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=None,
        ids_or_paths=["tile"],
        mode="none",
        s2_bands=[0, 2],
    )

    with pytest.raises(ValueError, match="out of range"):
        _ = ds[0]


def test_zscore_requires_matching_mean_std_lengths(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    img_root.mkdir()
    _write_tif(img_root / "tile.tif", np.ones((2, 2, 2), dtype=np.uint16))

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=None,
        ids_or_paths=["tile"],
        mode="none",
        normalize_cfg={"type": "zscore", "mean": [0.1], "std": [0.2]},
    )

    with pytest.raises(ValueError, match="Normalization stats length mismatch"):
        _ = ds[0]


def test_mask_is_binary_uint8_with_shape_1hw(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    mask_root = tmp_path / "mask"
    img_root.mkdir()
    mask_root.mkdir()
    _write_tif(img_root / "tile.tif", np.full((2, 2, 2), 2500, dtype=np.uint16))
    raw_mask = np.array([[0, 7], [2, 0]], dtype=np.int16)
    _write_tif(mask_root / "tile.tif", raw_mask)

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=mask_root,
        ids_or_paths=["tile"],
        mode="strong",
        reflectance_clip_percentile=0.0,
    )

    _img, mask, meta = ds[0]

    assert mask is not None
    assert mask.shape == (1, 2, 2)
    assert mask.dtype == torch.uint8
    assert set(torch.unique(mask).tolist()) == {0, 1}
    assert meta["modality"] == "optical"


def test_unlabeled_mode_returns_none_mask_and_boolean_valid_mask(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    img_root.mkdir()
    _write_tif(img_root / "tile.tif", np.full((3, 2, 2), 2500, dtype=np.uint16))

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=None,
        ids_or_paths=["tile"],
        mode="none",
        reflectance_clip_percentile=0.0,
    )

    img, mask, meta = ds[0]

    assert img.shape == (3, 2, 2)
    assert img.dtype == torch.float32
    assert torch.isfinite(img).all()
    assert mask is None
    assert meta["mask_path"] is None
    assert meta["valid_mask"].dtype == torch.bool
    assert tuple(meta["valid_mask"].shape) == (2, 2)


def test_resolves_direct_mask_path_before_suffix_mapping(tmp_path: Path) -> None:
    img_root = tmp_path / "img"
    mask_root = tmp_path / "mask"
    img_root.mkdir()
    mask_root.mkdir()
    _write_tif(img_root / "tile_A.tif", np.full((2, 2, 2), 2500, dtype=np.uint16))
    _write_tif(mask_root / "tile_A.tif", np.ones((2, 2), dtype=np.uint8))

    ds = OpticalDataset(
        img_root=img_root,
        mask_root=mask_root,
        ids_or_paths=["tile_A"],
        mode="strong",
        mask_id_suffix_map={"_A": "_B"},
    )

    assert Path(ds.samples[0]["mask_path"]).name == "tile_A.tif"
