from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import rasterio


@dataclass(frozen=True)
class CombinedManifestSample:
    sample_id: str
    manifest_index: int
    manifest_path: str
    sar_path: str
    optical_path: str
    label_path: Optional[str]


def load_combined_manifest_samples(
    combined_root_or_manifest: Union[str, Path],
    *,
    require_label: bool = False,
) -> List[CombinedManifestSample]:
    """
    Load canonical paired samples from the Combined manifest in row order.

    Each returned row is validated against the Combined directory layout and
    preserves the exact manifest ordering for strict paired fusion.
    """
    combined_root, manifest_path = _resolve_manifest_inputs(combined_root_or_manifest)

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_manifest_columns(
            reader.fieldnames,
            manifest_path,
            extra_required=("output_Label",) if require_label else (),
        )

        samples: List[CombinedManifestSample] = []
        for manifest_index, row in enumerate(reader):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(
                    f"Combined manifest row {manifest_index} in '{manifest_path}' is missing "
                    "'sample_id'"
                )

            sar_path = _resolve_manifest_asset(
                combined_root,
                row.get("output_S1", ""),
                expected_dir="S1",
                sample_id=sample_id,
                column="output_S1",
                manifest_index=manifest_index,
            )
            optical_path = _resolve_manifest_asset(
                combined_root,
                row.get("output_S2", ""),
                expected_dir="S2",
                sample_id=sample_id,
                column="output_S2",
                manifest_index=manifest_index,
            )

            label_path: Optional[str] = None
            label_cell = str(row.get("output_Label", "")).strip()
            if require_label:
                label_path = _resolve_manifest_asset(
                    combined_root,
                    label_cell,
                    expected_dir="Label",
                    sample_id=sample_id,
                    column="output_Label",
                    manifest_index=manifest_index,
                )
            elif label_cell:
                label_path = _resolve_manifest_asset(
                    combined_root,
                    label_cell,
                    expected_dir="Label",
                    sample_id=sample_id,
                    column="output_Label",
                    manifest_index=manifest_index,
                )

            samples.append(
                CombinedManifestSample(
                    sample_id=sample_id,
                    manifest_index=manifest_index,
                    manifest_path=str(manifest_path),
                    sar_path=sar_path,
                    optical_path=optical_path,
                    label_path=label_path,
                )
            )

    if not samples:
        raise ValueError(f"Combined manifest is empty: {manifest_path}")

    return samples


def validate_combined_manifest_samples(
    samples: Sequence[CombinedManifestSample],
    *,
    require_spatial_match: bool = True,
    require_label: bool = False,
) -> None:
    """Assert that every strict paired row is readable and jointly consistent."""
    for sample in samples:
        sar_hw = _read_raster_shape(sample.sar_path, sample_id=sample.sample_id, kind="SAR")
        optical_hw = _read_raster_shape(
            sample.optical_path,
            sample_id=sample.sample_id,
            kind="optical",
        )

        if require_spatial_match and sar_hw != optical_hw:
            raise ValueError(
                f"Combined manifest row {sample.manifest_index} for id='{sample.sample_id}' "
                f"has mismatched spatial dims: SAR {sar_hw} vs optical {optical_hw}"
            )

        if require_label:
            if sample.label_path is None:
                raise ValueError(
                    f"Combined manifest row {sample.manifest_index} for id='{sample.sample_id}' "
                    "is missing output_Label"
                )
            label_hw = _read_raster_shape(
                sample.label_path,
                sample_id=sample.sample_id,
                kind="label",
            )
            if sar_hw != label_hw:
                raise ValueError(
                    f"Combined manifest row {sample.manifest_index} for id='{sample.sample_id}' "
                    f"has mismatched SAR/label dims: SAR {sar_hw} vs label {label_hw}"
                )


def _resolve_manifest_inputs(
    combined_root_or_manifest: Union[str, Path]
) -> tuple[Path, Path]:
    path = Path(combined_root_or_manifest)
    if path.name.lower() == "manifest.csv":
        manifest_path = path
        combined_root = path.parent
    else:
        combined_root = path
        manifest_path = combined_root / "manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Combined manifest not found: {manifest_path}")
    if not combined_root.is_dir():
        raise FileNotFoundError(f"Combined root not found: {combined_root}")
    return combined_root, manifest_path


def _require_manifest_columns(
    fieldnames: Optional[Sequence[str]],
    manifest_path: Path,
    *,
    extra_required: Sequence[str] = (),
) -> None:
    if fieldnames is None:
        raise ValueError(f"Combined manifest has no header: {manifest_path}")

    required = {"sample_id", "output_S1", "output_S2", *extra_required}
    missing = sorted(name for name in required if name not in fieldnames)
    if missing:
        raise ValueError(
            f"Combined manifest '{manifest_path}' is missing required columns: {missing}"
        )


def _resolve_manifest_asset(
    combined_root: Path,
    raw_value: str,
    *,
    expected_dir: str,
    sample_id: str,
    column: str,
    manifest_index: int,
) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError(
            f"Combined manifest row {manifest_index} for id='{sample_id}' is missing "
            f"'{column}'"
        )

    rel_path = Path(value.replace("\\", "/"))
    if not rel_path.parts or rel_path.parts[0] != expected_dir:
        raise ValueError(
            f"Combined manifest row {manifest_index} for id='{sample_id}' has '{column}'="
            f"{value!r}, expected a path inside '{expected_dir}/'"
        )
    if rel_path.stem != sample_id:
        raise ValueError(
            f"Combined manifest row {manifest_index} for id='{sample_id}' has '{column}'="
            f"{value!r}, expected stem '{sample_id}'"
        )

    asset_path = combined_root / rel_path
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"Combined manifest row {manifest_index} for id='{sample_id}' references "
            f"missing {column} asset: {asset_path}"
        )
    return str(asset_path)


def _read_raster_shape(path: str, *, sample_id: str, kind: str) -> tuple[int, int]:
    try:
        with rasterio.open(path) as src:
            if src.count < 1:
                raise ValueError(
                    f"{kind} asset for id='{sample_id}' has zero bands at '{path}'"
                )
            return src.height, src.width
    except Exception as exc:
        raise ValueError(
            f"Failed to validate {kind} asset for id='{sample_id}' at '{path}': {exc}"
        ) from exc
