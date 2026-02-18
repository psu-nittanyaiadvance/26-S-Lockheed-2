from __future__ import annotations

import argparse
import csv
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds


DEFAULT_FILTERED_SUBDIR = Path("datasets") / "FilteredSouthAsia"
DEFAULT_OUTPUT = Path("data") / "derived" / "sen1floods11_time_manifest.csv"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_tif(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _find_datasets_root(repo_root: Path) -> Path:
    datasets_dir = repo_root / "datasets"
    if datasets_dir.exists():
        return datasets_dir
    data_dir = repo_root / "data"
    if data_dir.exists():
        return data_dir
    raise FileNotFoundError(
        f"Could not find datasets root. Expected {datasets_dir} or {data_dir}."
    )


def _find_metadata_files(search_root: Path) -> List[Path]:
    patterns = [
        "*sen1floods11*metadata*.csv",
        "*sen1floods11*.csv",
        "*SEN1Floods11*.csv",
        "*sen1floods11*metadata*.json",
        "*sen1floods11*.json",
        "*SEN1Floods11*.json",
    ]
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(search_root.rglob(pattern))
    return sorted({p for p in matches if p.is_file()})


def _pick_metadata_file(search_root: Path) -> Optional[Path]:
    candidates = _find_metadata_files(search_root)
    if not candidates:
        return None
    return candidates[0]


def _to_iso_date(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None


def _parse_bbox_from_row(row: pd.Series) -> Optional[Tuple[float, float, float, float]]:
    bbox_keys = [
        ("minlon", "minlat", "maxlon", "maxlat"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("lon_min", "lat_min", "lon_max", "lat_max"),
    ]
    lower = {c.lower(): c for c in row.index}
    for keys in bbox_keys:
        if all(k in lower for k in keys):
            try:
                vals = [float(row[lower[k]]) for k in keys]
                return vals[0], vals[1], vals[2], vals[3]
            except Exception:
                return None
    if "bbox" in lower:
        raw = row[lower["bbox"]]
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) == 4:
                try:
                    return tuple(float(p) for p in parts)  # type: ignore[return-value]
                except Exception:
                    return None
    return None


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Sen1Floods11 metadata to a minimal schema.

    Returned columns:
      - meta_id
      - event_id (optional)
      - s1_date
      - s2_date
      - bbox_wgs84 (optional)
      - orbit_direction (optional)
      - relative_orbit (optional)
    """
    id_col = _find_col(
        df,
        [
            "sample_id",
            "chip_id",
            "tile_id",
            "id",
            "im_id",
            "image_id",
            "patch_id",
            "flood_id",
        ],
    )
    if id_col is None:
        raise ValueError("Could not find a sample id column in metadata.")

    event_col = _find_col(df, ["event_id", "flood_event_id", "event", "flood_id", "storm_id"])
    s1_col = _find_col(df, ["s1_date", "s1_acq_date", "s1_acquisition", "s1_date_1", "s1_time"])
    s2_col = _find_col(df, ["s2_date", "s2_acq_date", "s2_acquisition", "s2_date_1", "s2_time"])
    orbit_dir_col = _find_col(df, ["orbit_direction", "orbit_pass", "s1_orbit_direction"])
    rel_orbit_col = _find_col(df, ["relative_orbit", "rel_orbit", "relativeOrbitNumber"])

    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        rec: Dict[str, object] = {
            "meta_id": str(row[id_col]),
            "event_id": str(row[event_col]) if event_col and pd.notna(row[event_col]) else None,
            "s1_date": _to_iso_date(row[s1_col]) if s1_col else None,
            "s2_date": _to_iso_date(row[s2_col]) if s2_col else None,
            "orbit_direction": str(row[orbit_dir_col]) if orbit_dir_col and pd.notna(row[orbit_dir_col]) else None,
            "relative_orbit": str(row[rel_orbit_col]) if rel_orbit_col and pd.notna(row[rel_orbit_col]) else None,
        }
        bbox = _parse_bbox_from_row(row)
        if bbox is not None:
            rec["bbox_wgs84"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        records.append(rec)

    return pd.DataFrame.from_records(records)


def _load_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".geojson"}:
        return pd.read_json(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


def _list_filtered_samples(filtered_root: Path) -> List[Dict[str, str]]:
    if not filtered_root.exists():
        raise FileNotFoundError(f"Filtered dataset root not found: {filtered_root}")

    split_map = {
        "WeaklyLabeled": "S1Weak",
        "HandLabeled": "S1Hand",
    }

    samples: List[Dict[str, str]] = []
    for split, s1_dir in split_map.items():
        img_dir = filtered_root / split / s1_dir
        if not img_dir.exists():
            warnings.warn(f"Missing image directory: {img_dir}")
            continue
        for path in img_dir.iterdir():
            if path.is_file() and _is_tif(path):
                samples.append(
                    {
                        "sample_id": path.stem,
                        "img_path": str(path),
                        "split": split,
                    }
                )
    if not samples:
        raise ValueError(f"No samples found under {filtered_root}")
    return samples


def _base_sample_id(sample_id: str) -> str:
    for suffix in ["_S1Weak", "_S1Hand", "_S2Weak", "_S2Hand", "_LabelHand", "_S1OtsuLabelWeak"]:
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def _bbox_from_raster(path: Path) -> Tuple[float, float, float, float]:
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("Missing CRS")
            bounds = src.bounds
            return transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21)
    except RasterioIOError as e:
        raise RuntimeError(f"Failed to read raster for bbox: {path}: {e}") from e


def build_time_manifest(
    filtered_root: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    repo_root = _repo_root()
    datasets_root = _find_datasets_root(repo_root)
    filtered_root = filtered_root or (repo_root / DEFAULT_FILTERED_SUBDIR)
    output_path = output_path or (repo_root / DEFAULT_OUTPUT)

    samples = _list_filtered_samples(filtered_root)

    if metadata_path is None:
        metadata_path = _pick_metadata_file(datasets_root)

    meta_df: Optional[pd.DataFrame] = None
    meta_by_id: Dict[str, Dict[str, object]] = {}

    if metadata_path is not None:
        raw_df = _load_metadata(metadata_path)
        meta_df = normalize_metadata(raw_df)
        for _, row in meta_df.iterrows():
            meta_by_id[str(row["meta_id"])] = row.to_dict()
    else:
        warnings.warn(
            "No metadata file found; s1_date/s2_date will be empty. "
            "Provide --metadata-path to include acquisition dates."
        )

    rows: List[Dict[str, object]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        base_id = _base_sample_id(sample_id)
        meta = meta_by_id.get(sample_id) or meta_by_id.get(base_id)

        bbox = None
        try:
            bbox = _bbox_from_raster(Path(sample["img_path"]))
        except Exception as exc:
            if meta and meta.get("bbox_wgs84"):
                bbox = tuple(float(x) for x in str(meta["bbox_wgs84"]).split(","))  # type: ignore[assignment]
            else:
                raise exc

        row: Dict[str, object] = {
            "sample_id": sample_id,
            "event_id": meta.get("event_id") if meta else None,
            "s1_date": meta.get("s1_date") if meta else None,
            "s2_date": meta.get("s2_date") if meta else None,
            "bbox_wgs84": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "split": sample["split"],
        }
        if meta:
            if meta.get("orbit_direction"):
                row["orbit_direction"] = meta.get("orbit_direction")
            if meta.get("relative_orbit"):
                row["relative_orbit"] = meta.get("relative_orbit")
        rows.append(row)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sen1Floods11 time manifest for GEE downloads.")
    parser.add_argument(
        "--filtered-root",
        type=Path,
        default=None,
        help="Path to filtered dataset root (default: datasets/FilteredSouthAsia).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Path to Sen1Floods11 metadata CSV/JSON. If omitted, search datasets folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/derived/sen1floods11_time_manifest.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_time_manifest(
        filtered_root=args.filtered_root,
        metadata_path=args.metadata_path,
        output_path=args.output,
    )
    print(f"Wrote {len(df)} rows to {args.output or DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
