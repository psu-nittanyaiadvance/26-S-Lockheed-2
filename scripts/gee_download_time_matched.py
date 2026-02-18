#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import rasterio

try:
    import ee
except Exception as exc:  # pragma: no cover
    ee = None
    _EE_IMPORT_ERROR = exc


DEFAULT_MANIFEST = Path("data") / "derived" / "sen1floods11_time_manifest.csv"
DEFAULT_OUTPUT_ROOT = Path("data") / "derived" / "gee_time_matched"
DEFAULT_OUTPUT_MANIFEST = Path("data") / "derived" / "gee_time_matched_manifest.csv"
DEFAULT_CONFIG = Path("configs") / "gee_time_match.yaml"


@dataclass
class DownloadResult:
    status: str
    s1_date_used: Optional[str]
    s2_date_used: Optional[str]
    window_s1: Optional[int]
    window_s2: Optional[int]
    path: Optional[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_value(raw: str):
    lowered = raw.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [item.strip() for item in inner.split(",")]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_config(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    current_list_key: Optional[str] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") and current_list_key:
            item = stripped[1:].strip()
            data[current_list_key].append(_parse_value(item))  # type: ignore[index]
            continue
        if ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val == "":
            data[key] = []
            current_list_key = key
        else:
            data[key] = _parse_value(val)
            current_list_key = None
    return data


def _ensure_ee() -> None:
    if ee is None:
        raise RuntimeError(
            "earthengine-api is not installed. Install it with `pip install earthengine-api`."
        )
    try:
        ee.Initialize()
    except Exception:
        raise RuntimeError(
            "Earth Engine is not authenticated. Run `earthengine authenticate` once, "
            "then retry."
        )


def _parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox string: {bbox_str}")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _expand_bbox(bbox: Tuple[float, float, float, float], factor: float) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bbox
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    half_w = (maxx - minx) / 2 * factor
    half_h = (maxy - miny) / 2 * factor
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _parse_date(value: str) -> datetime:
    dt = pd.to_datetime(value, errors="raise")
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    if isinstance(dt, datetime):
        return dt
    raise ValueError(f"Unrecognized date: {value}")


def _date_range(center: datetime, window_days: int) -> Tuple[str, str]:
    start = center - timedelta(days=window_days)
    end = center + timedelta(days=window_days)
    return start.date().isoformat(), end.date().isoformat()


def _filter_s1_collection(
    geom,
    s1_date: str,
    window_days: int,
    orbit_direction: Optional[str],
    relative_orbit: Optional[str],
):
    center = _parse_date(s1_date)
    start, end = _date_range(center, window_days)

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    if orbit_direction:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit_direction))
    if relative_orbit:
        try:
            rel_orbit_int = int(relative_orbit)
            col = col.filter(ee.Filter.eq("relativeOrbitNumber_start", rel_orbit_int))
        except ValueError:
            pass

    return col, start, end


def _filter_s2_collection(
    geom,
    s2_date: str,
    window_days: int,
    cloud_pct: float,
):
    center = _parse_date(s2_date)
    start, end = _date_range(center, window_days)

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    return col, start, end


def _try_collection(col, widen_days: int):
    count = col.size().getInfo()
    if count == 0:
        return None, 0, widen_days
    return col, count, None


def _download_geotiff(image, region, out_path: Path, scale: int, crs: Optional[str], max_pixels: int) -> None:
    params = {
        "scale": scale,
        "region": region,
        "fileFormat": "GeoTIFF",
        "format": "GEO_TIFF",
        "maxPixels": max_pixels,
    }
    if crs:
        params["crs"] = crs
    url = image.getDownloadURL(params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zip = Path(tmpdir) / "download.zip"
        urllib.request.urlretrieve(url, tmp_zip)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                raise RuntimeError("No GeoTIFF found in download zip.")
            zf.extract(tif_names[0], tmpdir)
            extracted = Path(tmpdir) / tif_names[0]
            extracted.replace(out_path)


def _validate_and_cast(path: Path, expected_bands: int = 8) -> None:
    with rasterio.open(path) as src:
        if src.count != expected_bands:
            raise ValueError(f"Expected {expected_bands} bands, got {src.count} in {path}")
        if src.crs is None or src.transform is None:
            raise ValueError(f"Missing CRS/transform in {path}")
        data = src.read()
        profile = src.profile

    if data.dtype != "float32":
        data = data.astype("float32")
        profile.update(dtype="float32")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)


def _safe_split(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "all"
    return str(value)


def _load_existing_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        return {}
    return {str(row["sample_id"]): row.to_dict() for _, row in df.iterrows()}


def _build_output_path(root: Path, split: str, sample_id: str) -> Path:
    return root / split / f"{sample_id}.tif"


def process_sample(
    row: pd.Series,
    cfg: Dict[str, object],
    output_root: Path,
    existing: Dict[str, Dict[str, str]],
) -> DownloadResult:
    sample_id = str(row["sample_id"])
    split = _safe_split(row.get("split"))

    output_path = _build_output_path(output_root, split, sample_id)
    prev = existing.get(sample_id)
    if prev and prev.get("status") == "ok" and output_path.exists():
        return DownloadResult(
            status="ok",
            s1_date_used=prev.get("s1_date_used"),
            s2_date_used=prev.get("s2_date_used"),
            window_s1=int(prev.get("window_s1")) if prev.get("window_s1") else None,
            window_s2=int(prev.get("window_s2")) if prev.get("window_s2") else None,
            path=str(output_path),
        )

    bbox_str = row.get("bbox_wgs84")
    if pd.isna(bbox_str):
        return DownloadResult("missing_both", None, None, None, None, None)

    bbox = _parse_bbox(str(bbox_str))
    if cfg.get("exclude_sen1floods11_exact_tiles"):
        factor = float(cfg.get("expand_bbox_factor", 2.0))
        bbox = _expand_bbox(bbox, factor)

    geom = ee.Geometry.Rectangle(list(bbox))

    s1_date = row.get("s1_date")
    s2_date = row.get("s2_date")
    if pd.isna(s1_date) and pd.isna(s2_date):
        return DownloadResult("missing_both", None, None, None, None, None)

    window_s1 = int(cfg.get("window_days_s1", 3))
    window_s2 = int(cfg.get("window_days_s2", 14))
    window_s1_wide = max(window_s1 * 2, 7)
    window_s2_wide = max(window_s2 * 2, 30)

    orbit_direction = row.get("orbit_direction") if "orbit_direction" in row else None
    relative_orbit = row.get("relative_orbit") if "relative_orbit" in row else None

    s1_img = None
    if not pd.isna(s1_date):
        col, _, _ = _filter_s1_collection(geom, str(s1_date), window_s1, orbit_direction, relative_orbit)
        if col.size().getInfo() == 0:
            col, _, _ = _filter_s1_collection(geom, str(s1_date), window_s1_wide, orbit_direction, relative_orbit)
            if col.size().getInfo() == 0:
                s1_img = None
                window_s1 = window_s1_wide
            else:
                s1_img = col.median()
                window_s1 = window_s1_wide
        else:
            s1_img = col.median()
    else:
        s1_img = None

    s2_img = None
    if not pd.isna(s2_date):
        cloud_pct = float(cfg.get("s2_cloud_pct", 20))
        col, _, _ = _filter_s2_collection(geom, str(s2_date), window_s2, cloud_pct)
        if col.size().getInfo() == 0:
            col, _, _ = _filter_s2_collection(geom, str(s2_date), window_s2_wide, cloud_pct)
            if col.size().getInfo() == 0:
                s2_img = None
                window_s2 = window_s2_wide
            else:
                s2_img = col.median()
                window_s2 = window_s2_wide
        else:
            s2_img = col.median()
    else:
        s2_img = None

    if s1_img is None and s2_img is None:
        return DownloadResult("missing_both", None, None, window_s1, window_s2, None)
    if s1_img is None:
        return DownloadResult("missing_s1", None, str(s2_date) if not pd.isna(s2_date) else None, window_s1, window_s2, None)
    if s2_img is None:
        return DownloadResult("missing_s2", str(s1_date) if not pd.isna(s1_date) else None, None, window_s1, window_s2, None)

    s2_bands = cfg.get("bands_s2", ["B2", "B3", "B4", "B8", "B11", "B12"])
    if isinstance(s2_bands, str):
        try:
            s2_bands = json.loads(s2_bands)
        except Exception:
            s2_bands = [b.strip() for b in s2_bands.split(",") if b.strip()]

    s1_proj = s1_img.select("VV").projection()
    crs = s1_proj.crs().getInfo()
    s2_resampled = s2_img.select(list(s2_bands)).resample("bilinear").reproject(s1_proj)

    stack = s1_img.select(["VV", "VH"]).addBands(s2_resampled)

    scale = int(cfg.get("export_scale_m", 10))
    max_pixels = int(cfg.get("max_pixels", 1_000_000_00))

    _download_geotiff(stack, geom.toGeoJSONString(), output_path, scale=scale, crs=crs, max_pixels=max_pixels)
    _validate_and_cast(output_path)

    return DownloadResult(
        status="ok",
        s1_date_used=str(s1_date),
        s2_date_used=str(s2_date),
        window_s1=window_s1,
        window_s2=window_s2,
        path=str(output_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download time-matched S1/S2 imagery from GEE.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config YAML.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Time manifest CSV.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root directory.")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
        help="Output manifest CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config.exists() else {}

    _ensure_ee()

    df = pd.read_csv(args.manifest)
    split_filter = cfg.get("split_filter")
    if split_filter:
        if isinstance(split_filter, list):
            df = df[df["split"].astype(str).isin([str(s) for s in split_filter])]
        else:
            df = df[df["split"].astype(str) == str(split_filter)]

    existing = _load_existing_manifest(args.output_manifest)

    results: List[Dict[str, object]] = []
    counts = {"ok": 0, "missing_s1": 0, "missing_s2": 0, "missing_both": 0, "error": 0}

    for _, row in df.iterrows():
        sample_id = str(row["sample_id"])
        try:
            res = process_sample(row, cfg, args.output_root, existing)
            counts[res.status] = counts.get(res.status, 0) + 1
            results.append(
                {
                    "sample_id": sample_id,
                    "path": res.path,
                    "status": res.status,
                    "s1_date_used": res.s1_date_used,
                    "s2_date_used": res.s2_date_used,
                    "window_s1": res.window_s1,
                    "window_s2": res.window_s2,
                }
            )
            print(f"{sample_id}: {res.status}")
        except Exception as exc:
            counts["error"] += 1
            results.append(
                {
                    "sample_id": sample_id,
                    "path": None,
                    "status": "error",
                    "s1_date_used": row.get("s1_date"),
                    "s2_date_used": row.get("s2_date"),
                    "window_s1": cfg.get("window_days_s1"),
                    "window_s2": cfg.get("window_days_s2"),
                }
            )
            print(f"{sample_id}: error -> {exc}")

    out_df = pd.DataFrame(results)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_manifest, index=False)

    print("Summary:")
    for key, val in counts.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
