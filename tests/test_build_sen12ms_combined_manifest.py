from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader.combined_manifest import load_combined_manifest_samples  # noqa: E402


def _load_builder_module():
    script_path = ROOT / "scripts" / "build_sen12ms_combined_manifest.py"
    module_name = "build_sen12ms_combined_manifest"
    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def _write_raw_pair(
    raw_root: Path,
    *,
    roi: str,
    season: str,
    inner_index: str,
    patch_id: str,
) -> None:
    sar_root = raw_root / f"{roi}_{season}_s1" / f"s1_{inner_index}"
    optical_root = raw_root / f"{roi}_{season}_s2" / f"s2_{inner_index}"
    cloudy_root = raw_root / f"{roi}_{season}_s2_cloudy" / f"s2_cloudy_{inner_index}"

    _write_tif(
        sar_root / f"{roi}_{season}_s1_{inner_index}_{patch_id}.tif",
        np.ones((2, 4, 4), dtype=np.float32),
    )
    _write_tif(
        optical_root / f"{roi}_{season}_s2_{inner_index}_{patch_id}.tif",
        np.full((3, 4, 4), 2500, dtype=np.uint16),
    )
    _write_tif(
        cloudy_root / f"{roi}_{season}_s2_cloudy_{inner_index}_{patch_id}.tif",
        np.full((3, 4, 4), 999, dtype=np.uint16),
    )


def test_build_combined_dataset_writes_strict_manifest_and_validates(
    tmp_path: Path,
) -> None:
    module = _load_builder_module()
    raw_root = tmp_path / "datasets" / "SEN12MS"
    combined_root = raw_root / "Combined"

    _write_raw_pair(
        raw_root,
        roi="ROIs1970",
        season="fall",
        inner_index="2",
        patch_id="p30",
    )
    _write_raw_pair(
        raw_root,
        roi="ROIs1158",
        season="spring",
        inner_index="1",
        patch_id="p7",
    )

    stats = module.build_combined_dataset(
        raw_root=raw_root,
        combined_root=combined_root,
        link_mode="copy",
        validate=True,
    )

    manifest_path = combined_root / "manifest.csv"
    assert manifest_path.is_file()
    assert (combined_root / "S1").is_dir()
    assert (combined_root / "S2").is_dir()

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["sample_id", "output_S1", "output_S2"]
        rows = list(reader)

    assert [row["sample_id"] for row in rows] == [
        "ROIs1158_spring_1_p7",
        "ROIs1970_fall_2_p30",
    ]
    assert rows[0]["output_S1"] == "S1/ROIs1158_spring_1_p7.tif"
    assert rows[0]["output_S2"] == "S2/ROIs1158_spring_1_p7.tif"
    assert rows[1]["output_S1"] == "S1/ROIs1970_fall_2_p30.tif"
    assert rows[1]["output_S2"] == "S2/ROIs1970_fall_2_p30.tif"

    for row in rows:
        sample_id = row["sample_id"]
        assert (combined_root / row["output_S1"]).stem == sample_id
        assert (combined_root / row["output_S2"]).stem == sample_id

    samples = load_combined_manifest_samples(combined_root)
    assert [sample.sample_id for sample in samples] == [
        "ROIs1158_spring_1_p7",
        "ROIs1970_fall_2_p30",
    ]

    assert stats.sar_files_discovered == 2
    assert stats.optical_files_discovered == 2
    assert stats.matched_pairs_written == 2
    assert stats.unmatched_sar_files == 0
    assert stats.unmatched_optical_files == 0
    assert stats.materialization_mode == "copy"
    assert stats.validation_ran is True


def test_build_combined_dataset_rejects_unmatched_tiles(tmp_path: Path) -> None:
    module = _load_builder_module()
    raw_root = tmp_path / "datasets" / "SEN12MS"

    sar_path = (
        raw_root
        / "ROIs1158_spring_s1"
        / "s1_1"
        / "ROIs1158_spring_s1_1_p30.tif"
    )
    optical_root = raw_root / "ROIs1158_spring_s2"
    optical_root.mkdir(parents=True, exist_ok=True)
    _write_tif(sar_path, np.ones((2, 4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="unmatched SAR files: 1"):
        module.build_combined_dataset(
            raw_root=raw_root,
            combined_root=raw_root / "Combined",
            link_mode="copy",
        )


def test_build_combined_dataset_rejects_existing_nonempty_output_without_clean(
    tmp_path: Path,
) -> None:
    module = _load_builder_module()
    raw_root = tmp_path / "datasets" / "SEN12MS"
    combined_root = raw_root / "Combined"

    _write_raw_pair(
        raw_root,
        roi="ROIs1158",
        season="spring",
        inner_index="1",
        patch_id="p30",
    )
    _write_tif(
        combined_root / "S1" / "stale_file.tif",
        np.ones((1, 2, 2), dtype=np.uint8),
    )

    with pytest.raises(FileExistsError, match="Output modality directory is not empty"):
        module.build_combined_dataset(
            raw_root=raw_root,
            combined_root=combined_root,
            link_mode="copy",
        )


def test_build_combined_dataset_rejects_bad_filename_parse(tmp_path: Path) -> None:
    module = _load_builder_module()
    raw_root = tmp_path / "datasets" / "SEN12MS"

    bad_sar = raw_root / "ROIs1158_spring_s1" / "s1_1" / "bad_name.tif"
    good_s2 = (
        raw_root
        / "ROIs1158_spring_s2"
        / "s2_1"
        / "ROIs1158_spring_s2_1_p30.tif"
    )
    _write_tif(bad_sar, np.ones((2, 4, 4), dtype=np.float32))
    _write_tif(good_s2, np.full((3, 4, 4), 2500, dtype=np.uint16))

    with pytest.raises(ValueError, match="Failed to parse raw SEN12MS TIFF filename"):
        module.build_combined_dataset(
            raw_root=raw_root,
            combined_root=raw_root / "Combined",
            link_mode="copy",
        )
