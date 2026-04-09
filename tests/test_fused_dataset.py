from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import FusedDataset, PatchDataset, multimodal_pretrain_collate  # noqa: E402
from data_loader.combined_manifest import CombinedManifestSample  # noqa: E402


class _TupleDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]],
        *,
        with_samples: bool = True,
        normalize_type: Optional[str] = None,
    ) -> None:
        self.items = list(items)
        if with_samples:
            self.samples = [
                {
                    "id": meta["id"],
                    "img_path": meta.get("img_path"),
                    "mask_path": meta.get("mask_path"),
                }
                for _, _, meta in self.items
            ]
        if normalize_type is not None:
            self._normalize_type = normalize_type

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        return self.items[idx]


def _make_item(
    sample_id: str,
    *,
    channels: int,
    hw: Tuple[int, int] = (2, 2),
    img_value: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    modality: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    img = torch.full((channels, *hw), img_value, dtype=torch.float32)
    if valid_mask is None:
        valid_mask = torch.ones(hw, dtype=torch.bool)
    meta: Dict[str, Any] = {
        "id": sample_id,
        "img_path": f"{sample_id}.tif",
        "mask_path": f"{sample_id}_mask.tif" if mask is not None else None,
        "valid_mask": valid_mask,
    }
    if modality is not None:
        meta["modality"] = modality
    return img, mask, meta


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
    manifest_rows: Sequence[Dict[str, str]],
    sar_shape: Tuple[int, int] = (4, 4),
    optical_shape: Tuple[int, int] = (4, 4),
    include_labels: bool = True,
    missing_assets: Optional[set[Tuple[str, str]]] = None,
    optical_value: int = 2500,
) -> Path:
    combined_root = tmp_path / "Combined"
    missing_assets = missing_assets or set()

    for row in manifest_rows:
        sample_id = row["sample_id"]
        if ("S1", sample_id) not in missing_assets:
            _write_tif(
                combined_root / "S1" / f"{sample_id}.tif",
                np.ones((2, *sar_shape), dtype=np.float32),
            )
        if ("S2", sample_id) not in missing_assets:
            _write_tif(
                combined_root / "S2" / f"{sample_id}.tif",
                np.full((3, *optical_shape), optical_value, dtype=np.uint16),
            )
        if include_labels and ("Label", sample_id) not in missing_assets:
            _write_tif(
                combined_root / "Label" / f"{sample_id}.tif",
                np.ones(sar_shape, dtype=np.uint8),
            )

    header = [
        "sample_id",
        "output_S1",
        "output_S2",
        "output_Label",
    ]
    lines = [",".join(header)]
    for row in manifest_rows:
        values = [row.get(column, "") for column in header]
        lines.append(",".join(values))
    manifest_path = combined_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return combined_root


def test_len_matches_sar_dataset() -> None:
    sar_ds = _TupleDataset(
        [
            _make_item("tile_1", channels=2),
            _make_item("tile_2", channels=2),
        ]
    )
    opt_ds = _TupleDataset([_make_item("tile_1", channels=4, modality="optical")])

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")

    assert len(fused) == len(sar_ds)


def test_raise_policy_rejects_missing_ids_at_init_when_samples_attr_exists() -> None:
    sar_ds = _TupleDataset([_make_item("sar_only", channels=2)])
    opt_ds = _TupleDataset([_make_item("other_id", channels=4, modality="optical")])

    with pytest.raises(ValueError, match="have no paired optical tile"):
        FusedDataset(sar_ds, opt_ds, optical_missing_policy="raise")


def test_getitem_fuses_channels_and_preserves_mask() -> None:
    sar_mask = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.uint8)
    sar_ds = _TupleDataset(
        [_make_item("tile", channels=2, img_value=1.0, mask=sar_mask)]
    )
    opt_ds = _TupleDataset(
        [_make_item("tile", channels=3, img_value=2.0, modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds)
    img, mask, meta = fused[0]

    assert img.shape == (5, 2, 2)
    assert torch.equal(img[:2], sar_ds[0][0])
    assert torch.equal(img[2:], opt_ds[0][0])
    assert mask is not None
    assert torch.equal(mask, sar_mask)
    assert meta["fused_optical_available"] is True
    assert meta["modality"] == "fused"
    assert meta["img_path"] == "tile.tif"
    assert meta["mask_path"] == "tile_mask.tif"


def test_fused_valid_mask_is_logical_and_of_sar_and_optical() -> None:
    sar_valid = torch.tensor([[True, True], [False, True]])
    opt_valid = torch.tensor([[True, False], [True, True]])
    sar_ds = _TupleDataset(
        [_make_item("tile", channels=2, valid_mask=sar_valid)]
    )
    opt_ds = _TupleDataset(
        [_make_item("tile", channels=3, valid_mask=opt_valid, modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds)
    _img, _mask, meta = fused[0]

    assert torch.equal(meta["valid_mask"], sar_valid & opt_valid)


def test_missing_optical_with_zeros_policy_zero_fills_and_preserves_sar_valid_mask() -> None:
    sar_mask = torch.ones((1, 2, 2), dtype=torch.uint8)
    sar_valid = torch.tensor([[True, False], [True, True]])
    sar_ds = _TupleDataset(
        [_make_item("missing_tile", channels=2, mask=sar_mask, valid_mask=sar_valid)]
    )
    opt_ds = _TupleDataset(
        [_make_item("other_tile", channels=4, modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")
    img, mask, meta = fused[0]

    assert img.shape == (6, 2, 2)
    assert torch.equal(img[:2], sar_ds[0][0])
    assert torch.equal(img[2:], torch.zeros((4, 2, 2), dtype=torch.float32))
    assert mask is not None
    assert torch.equal(mask, sar_mask)
    assert meta["fused_optical_available"] is False
    assert meta["modality"] == "fused"
    assert meta["img_path"] == "missing_tile.tif"
    assert meta["mask_path"] == "missing_tile_mask.tif"
    assert meta["n_sar_bands"] == 2
    assert meta["n_optical_bands"] == 4
    assert torch.equal(meta["valid_mask"], sar_valid)


def test_missing_optical_with_raise_policy_raises_at_getitem_without_samples_attr() -> None:
    sar_ds = _TupleDataset([_make_item("missing_tile", channels=2)], with_samples=False)
    opt_ds = _TupleDataset([_make_item("other_tile", channels=3, modality="optical")])

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="raise")

    with pytest.raises(RuntimeError, match="No paired optical tile"):
        _ = fused[0]


def test_spatial_mismatch_raises_when_required() -> None:
    sar_ds = _TupleDataset([_make_item("tile", channels=2, hw=(2, 2))])
    opt_ds = _TupleDataset(
        [_make_item("tile", channels=3, hw=(3, 2), modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds, require_spatial_match=True)

    with pytest.raises(ValueError, match="spatial dims differ"):
        _ = fused[0]


def test_channel_count_metadata_matches_fused_image() -> None:
    sar_ds = _TupleDataset([_make_item("tile", channels=2)])
    opt_ds = _TupleDataset([_make_item("tile", channels=4, modality="optical")])

    fused = FusedDataset(sar_ds, opt_ds)
    img, _mask, meta = fused[0]

    assert meta["n_sar_bands"] == 2
    assert meta["n_optical_bands"] == 4
    assert img.shape[0] == meta["n_sar_bands"] + meta["n_optical_bands"]


def test_unlabeled_sar_sample_keeps_mask_none_even_if_optical_has_mask() -> None:
    opt_mask = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8)
    sar_ds = _TupleDataset([_make_item("tile", channels=2, mask=None)])
    opt_ds = _TupleDataset(
        [_make_item("tile", channels=4, mask=opt_mask, modality="optical")]
    )

    fused = FusedDataset(sar_ds, opt_ds)
    _img, mask, meta = fused[0]

    assert mask is None
    assert meta["mask_path"] is None


def test_repr_reports_counts_and_policy() -> None:
    sar_ds = _TupleDataset(
        [_make_item("tile_1", channels=2), _make_item("tile_2", channels=2)]
    )
    opt_ds = _TupleDataset([_make_item("tile_1", channels=3, modality="optical")])

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")
    text = repr(fused)

    assert "n_sar=2" in text
    assert "n_optical=1" in text
    assert "n_paired=1" in text
    assert "policy='zeros'" in text


def test_infer_optical_bands_falls_back_to_one_when_optical_dataset_empty() -> None:
    sar_ds = _TupleDataset([_make_item("tile", channels=2)])
    opt_ds = _TupleDataset([], with_samples=True)

    fused = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")
    img, _mask, meta = fused[0]

    assert img.shape == (3, 2, 2)
    assert meta["n_optical_bands"] == 1
    assert torch.equal(img[2:], torch.zeros((1, 2, 2), dtype=torch.float32))


def test_sar_only_is_behaviorally_identical_to_zeros_alias() -> None:
    sar_ds = _TupleDataset([_make_item("missing_tile", channels=2)])
    opt_ds = _TupleDataset([_make_item("other_tile", channels=3, modality="optical")])

    zeros_ds = FusedDataset(sar_ds, opt_ds, optical_missing_policy="zeros")
    sar_only_ds = FusedDataset(sar_ds, opt_ds, optical_missing_policy="sar_only")

    zeros_img, zeros_mask, zeros_meta = zeros_ds[0]
    sar_only_img, sar_only_mask, sar_only_meta = sar_only_ds[0]

    assert torch.equal(zeros_img, sar_only_img)
    assert zeros_mask == sar_only_mask
    assert torch.equal(zeros_meta["valid_mask"], sar_only_meta["valid_mask"])
    assert zeros_meta["fused_optical_available"] == sar_only_meta["fused_optical_available"]
    assert zeros_meta["n_optical_bands"] == sar_only_meta["n_optical_bands"]
    assert zeros_meta["optical_img_path"] == sar_only_meta["optical_img_path"]
    assert zeros_ds.optical_missing_policy == "zeros"
    assert sar_only_ds.optical_missing_policy == "zeros"


def test_strict_from_combined_manifest_preserves_manifest_order_not_directory_order(
    tmp_path: Path,
) -> None:
    combined_root = _build_combined_root(
        tmp_path,
        manifest_rows=[
            {
                "sample_id": "tile_b",
                "output_S1": "S1\\tile_b.tif",
                "output_S2": "S2\\tile_b.tif",
                "output_Label": "Label\\tile_b.tif",
            },
            {
                "sample_id": "tile_a",
                "output_S1": "S1\\tile_a.tif",
                "output_S2": "S2\\tile_a.tif",
                "output_Label": "Label\\tile_a.tif",
            },
        ],
    )

    fused = FusedDataset.from_combined_manifest(combined_root, mode="none")

    assert [sample["id"] for sample in fused.sar_dataset.samples] == ["tile_b", "tile_a"]
    assert [sample["id"] for sample in fused.optical_dataset.samples] == ["tile_b", "tile_a"]
    assert [fused[i][2]["id"] for i in range(len(fused))] == ["tile_b", "tile_a"]


def test_strict_from_combined_manifest_enforces_exact_parity_and_metadata(
    tmp_path: Path,
) -> None:
    combined_root = _build_combined_root(
        tmp_path,
        manifest_rows=[
            {
                "sample_id": "tile_1",
                "output_S1": "S1\\tile_1.tif",
                "output_S2": "S2\\tile_1.tif",
                "output_Label": "Label\\tile_1.tif",
            },
            {
                "sample_id": "tile_2",
                "output_S1": "S1\\tile_2.tif",
                "output_S2": "S2\\tile_2.tif",
                "output_Label": "Label\\tile_2.tif",
            },
        ],
    )

    fused = FusedDataset.from_combined_manifest(combined_root, mode="none")
    _img, mask, meta = fused[1]

    sar_ids = [sample["id"] for sample in fused.sar_dataset.samples]
    optical_ids = [sample["id"] for sample in fused.optical_dataset.samples]

    assert len(sar_ids) == len(optical_ids) == len(fused) == 2
    assert sar_ids == optical_ids == ["tile_1", "tile_2"]
    assert mask is None
    assert meta["paired_sample_id"] == "tile_2"
    assert meta["strict_paired_mode"] is True
    assert meta["pairing_source"] == "combined_manifest"
    assert Path(meta["sar_img_path"]).name == "tile_2.tif"
    assert Path(meta["optical_img_path"]).name == "tile_2.tif"
    assert Path(meta["label_path"]).name == "tile_2.tif"
    assert meta["manifest_row_index"] == 1


def test_strict_from_combined_manifest_requires_optical_assets(tmp_path: Path) -> None:
    combined_root = _build_combined_root(
        tmp_path,
        manifest_rows=[
            {
                "sample_id": "tile_1",
                "output_S1": "S1\\tile_1.tif",
                "output_S2": "S2\\tile_1.tif",
                "output_Label": "Label\\tile_1.tif",
            }
        ],
        missing_assets={("S2", "tile_1")},
    )

    with pytest.raises(FileNotFoundError, match="missing output_S2 asset"):
        FusedDataset.from_combined_manifest(combined_root, mode="none")


def test_strict_from_combined_manifest_requires_manifest_columns(tmp_path: Path) -> None:
    combined_root = tmp_path / "Combined"
    combined_root.mkdir(parents=True, exist_ok=True)
    (combined_root / "manifest.csv").write_text(
        "sample_id,output_S1\n"
        "tile_1,S1\\\\tile_1.tif\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        FusedDataset.from_combined_manifest(combined_root, mode="none")


def test_strict_from_combined_manifest_rejects_independent_validate_flags(
    tmp_path: Path,
) -> None:
    combined_root = _build_combined_root(
        tmp_path,
        manifest_rows=[
            {
                "sample_id": "tile_1",
                "output_S1": "S1\\tile_1.tif",
                "output_S2": "S2\\tile_1.tif",
                "output_Label": "Label\\tile_1.tif",
            }
        ],
    )

    with pytest.raises(ValueError, match="validates jointly"):
        FusedDataset.from_combined_manifest(
            combined_root,
            mode="none",
            optical_dataset_kwargs={"validate": False},
        )


def test_direct_strict_pairing_rejects_zero_fallback_policy() -> None:
    sar_ds = _TupleDataset([_make_item("tile", channels=2)])
    opt_ds = _TupleDataset([_make_item("tile", channels=3, modality="optical")])
    strict_manifest_samples = [
        CombinedManifestSample(
            sample_id="tile",
            manifest_index=0,
            manifest_path="Combined/manifest.csv",
            sar_path="tile.tif",
            optical_path="tile.tif",
            label_path=None,
        )
    ]

    with pytest.raises(ValueError, match="requires optical_missing_policy='raise'"):
        FusedDataset(
            sar_ds,
            opt_ds,
            optical_missing_policy="zeros",
            strict_pairing=True,
            strict_manifest_samples=strict_manifest_samples,
        )


def test_strict_from_combined_manifest_validation_raises_instead_of_silent_drift(
    tmp_path: Path,
) -> None:
    combined_root = _build_combined_root(
        tmp_path,
        manifest_rows=[
            {
                "sample_id": "tile_bad",
                "output_S1": "S1\\tile_bad.tif",
                "output_S2": "S2\\tile_bad.tif",
                "output_Label": "Label\\tile_bad.tif",
            }
        ],
        sar_shape=(4, 4),
        optical_shape=(5, 4),
    )

    with pytest.raises(ValueError, match="mismatched spatial dims"):
        FusedDataset.from_combined_manifest(
            combined_root,
            mode="none",
            validate=True,
        )


def test_strict_valid_mask_is_joint_intersection_when_both_modalities_exist() -> None:
    sar_valid = torch.tensor([[True, True], [False, True]])
    opt_valid = torch.tensor([[True, False], [True, True]])
    strict_manifest_samples = [
        CombinedManifestSample(
            sample_id="tile",
            manifest_index=0,
            manifest_path="Combined/manifest.csv",
            sar_path="Combined/S1/tile.tif",
            optical_path="Combined/S2/tile.tif",
            label_path=None,
        )
    ]
    sar_img, sar_mask, sar_meta = _make_item("tile", channels=2, valid_mask=sar_valid)
    sar_meta["img_path"] = "Combined/S1/tile.tif"
    opt_img, opt_mask, opt_meta = _make_item(
        "tile",
        channels=3,
        valid_mask=opt_valid,
        modality="optical",
    )
    opt_meta["img_path"] = "Combined/S2/tile.tif"
    sar_ds = _TupleDataset([(sar_img, sar_mask, sar_meta)])
    opt_ds = _TupleDataset([(opt_img, opt_mask, opt_meta)])

    fused = FusedDataset(
        sar_ds,
        opt_ds,
        strict_pairing=True,
        strict_manifest_samples=strict_manifest_samples,
    )
    _img, _mask, meta = fused[0]

    assert meta["strict_paired_mode"] is True
    assert torch.equal(meta["valid_mask"], sar_valid & opt_valid)


def test_strict_patched_fused_samples_keep_deterministic_patch_provenance(
    tmp_path: Path,
) -> None:
    combined_root = _build_combined_root(
        tmp_path,
        manifest_rows=[
            {
                "sample_id": "tile_1",
                "output_S1": "S1\\tile_1.tif",
                "output_S2": "S2\\tile_1.tif",
                "output_Label": "Label\\tile_1.tif",
            }
        ],
    )

    fused = FusedDataset.from_combined_manifest(combined_root, mode="none")
    patched = PatchDataset(fused, patch_size=2, overlap=0.0)
    _img, mask, meta = patched[0]

    assert mask is None
    assert meta["base_id"] == "tile_1"
    assert meta["id"] == "tile_1_r0_c0"
    assert meta["strict_paired_mode"] is True
    assert meta["paired_sample_id"] == "tile_1"
    assert Path(meta["sar_img_path"]).name == "tile_1.tif"
    assert Path(meta["optical_img_path"]).name == "tile_1.tif"
    assert tuple(meta["valid_mask"].shape) == (2, 2)
