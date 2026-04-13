from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import (  # noqa: E402
    CachedPairedDataset,
    FusedDataset,
    build_paired_preprocessed_cache,
    build_paired_preprocessing_config,
    multimodal_pretrain_collate,
    validate_paired_cache_parity,
)


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


def _build_combined_root(
    tmp_path: Path,
    *,
    sample_ids: Sequence[str] = ("tile_b", "tile_a"),
    hw: Tuple[int, int] = (4, 4),
) -> Path:
    combined_root = tmp_path / "Combined"
    h, w = hw
    base = np.arange(h * w, dtype=np.float32).reshape(h, w) + 1.0

    lines = ["sample_id,output_S1,output_S2,output_Label"]
    for idx, sample_id in enumerate(sample_ids):
        sar = np.stack([base + idx, base * 2.0 + idx], axis=0).astype(np.float32)
        if idx == 0:
            sar[0, 0, 0] = np.nan
        optical = np.stack(
            [
                (base + band + idx).astype(np.uint16) * 100
                for band in range(13)
            ],
            axis=0,
        )
        if idx == 1:
            optical[:, 1, 1] = 0

        _write_tif(combined_root / "S1" / f"{sample_id}.tif", sar)
        _write_tif(combined_root / "S2" / f"{sample_id}.tif", optical)
        _write_tif(combined_root / "Label" / f"{sample_id}.tif", np.ones(hw, dtype=np.uint8))
        lines.append(
            f"{sample_id},S1\\{sample_id}.tif,S2\\{sample_id}.tif,Label\\{sample_id}.tif"
        )

    (combined_root / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return combined_root


def _build_live_dataset(combined_root: Path) -> FusedDataset:
    return FusedDataset.from_combined_manifest(
        combined_root,
        mode="none",
        return_mode="paired",
        sar_dataset_kwargs={"log_transform": False, "expected_img_bands": 2},
        optical_dataset_kwargs={
            "expected_img_bands": 13,
            "reflectance_clip_percentile": 2.0,
        },
        require_spatial_match=True,
        validate=True,
    )


def _cache_config(combined_root: Path) -> Dict[str, object]:
    return build_paired_preprocessing_config(
        combined_root,
        sar_dataset_kwargs={"log_transform": False, "expected_img_bands": 2},
        optical_dataset_kwargs={
            "expected_img_bands": 13,
            "reflectance_clip_percentile": 2.0,
        },
        require_spatial_match=True,
        validate_manifest=True,
    )


def test_build_cache_subset_and_cached_sample_contract_matches_live(tmp_path: Path) -> None:
    combined_root = _build_combined_root(tmp_path)
    live = _build_live_dataset(combined_root)
    cache_dir = tmp_path / "paired_cache"

    index = build_paired_preprocessed_cache(
        live,
        cache_dir,
        preprocessing_config=_cache_config(combined_root),
        max_samples=1,
    )
    cached = CachedPairedDataset(
        cache_dir,
        expected_preprocessing_config_digest=index["preprocessing_config_digest"],
    )
    live_sample = live[0]
    cached_sample = cached[0]

    assert len(cached) == 1
    assert set(cached_sample) == {"sar", "optical", "valid_mask", "meta"}
    assert torch.equal(cached_sample["sar"], live_sample["sar"])
    assert torch.equal(cached_sample["optical"], live_sample["optical"])
    assert torch.equal(cached_sample["valid_mask"], live_sample["valid_mask"])
    assert cached_sample["meta"]["paired_sample_id"] == live_sample["meta"]["paired_sample_id"]
    assert cached_sample["meta"]["manifest_row_index"] == live_sample["meta"]["manifest_row_index"]
    assert "preprocessing_config_digest" in cached_sample["meta"]


def test_cache_persists_audit_masks_and_parity_is_exact(tmp_path: Path) -> None:
    combined_root = _build_combined_root(tmp_path)
    live = _build_live_dataset(combined_root)
    cache_dir = tmp_path / "paired_cache"

    config = _cache_config(combined_root)
    build_paired_preprocessed_cache(live, cache_dir, preprocessing_config=config)
    cached = CachedPairedDataset(
        cache_dir,
        expected_preprocessing_config_digest=config["preprocessing_config_digest"],
    )

    parity = validate_paired_cache_parity(live, cached)
    cached_record_0 = cached.load_record(0)
    live_pair_0 = live.get_paired_item(0, require_no_transforms=True)

    assert parity == {"checked": 2, "ok": True, "mismatches": [], "atol": 0.0, "rtol": 0.0}
    assert torch.equal(cached_record_0["sar_valid_mask"], live_pair_0["metadata"]["sar_valid_mask"])
    assert torch.equal(
        cached_record_0["optical_valid_mask"],
        live_pair_0["metadata"]["optical_valid_mask"],
    )


def test_cached_dataset_is_collate_compatible_and_preserves_manifest_order(tmp_path: Path) -> None:
    combined_root = _build_combined_root(tmp_path, sample_ids=("tile_z", "tile_m", "tile_a"))
    live = _build_live_dataset(combined_root)
    cache_dir = tmp_path / "paired_cache"

    build_paired_preprocessed_cache(live, cache_dir, preprocessing_config=_cache_config(combined_root))
    cached = CachedPairedDataset(cache_dir)
    loader = DataLoader(cached, batch_size=2, shuffle=False, collate_fn=multimodal_pretrain_collate)
    batch = next(iter(loader))

    assert [cached[i]["meta"]["paired_sample_id"] for i in range(len(cached))] == [
        "tile_z",
        "tile_m",
        "tile_a",
    ]
    assert tuple(batch["sar"].shape) == (2, 2, 4, 4)
    assert tuple(batch["optical"].shape) == (2, 13, 4, 4)
    assert tuple(batch["valid_mask"].shape) == (2, 1, 4, 4)
    assert [meta["paired_sample_id"] for meta in batch["meta"]] == ["tile_z", "tile_m"]
