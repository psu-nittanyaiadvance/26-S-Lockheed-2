from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


ROOT_NAME_RE = re.compile(r"^(ROIs\d+)_([a-z0-9]+)_(s1|s2|s2_cloudy)$")
FILE_NAME_RE = re.compile(
    r"^(ROIs\d+)_([a-z0-9]+)_(s1|s2|s2_cloudy)_(\d+)_(p\d+)$"
)
RASTER_SUFFIXES = {".tif", ".tiff"}
MANIFEST_COLUMNS = ("sample_id", "output_S1", "output_S2")
EXPECTED_MODALITIES = ("s1", "s2")


@dataclass(frozen=True, order=True)
class SampleKey:
    roi_prefix: str
    season: str
    inner_index: str
    patch_id: str

    @property
    def sample_id(self) -> str:
        return (
            f"{self.roi_prefix}_{self.season}_{self.inner_index}_{self.patch_id}"
        )


@dataclass(frozen=True)
class RootSpec:
    roi_prefix: str
    season: str
    modality: str
    path: Path

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.roi_prefix, self.season)


@dataclass(frozen=True)
class TileRecord:
    key: SampleKey
    modality: str
    path: Path


@dataclass(frozen=True)
class BuildStats:
    sar_files_discovered: int
    optical_files_discovered: int
    matched_pairs_written: int
    unmatched_sar_files: int
    unmatched_optical_files: int
    materialization_mode: str
    validation_ran: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict Combined manifest dataset for SEN12MS from the raw "
            "season/modality folder layout."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets/SEN12MS"),
        help="Root containing raw SEN12MS season/modality directories.",
    )
    parser.add_argument(
        "--combined-root",
        type=Path,
        default=Path("datasets/SEN12MS/Combined"),
        help="Output Combined root where manifest.csv, S1/, and S2/ will be written.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize Combined assets. Symlink is the default.",
    )
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Allow replacing an existing manifest.csv if output asset directories are empty.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing Combined/S1, Combined/S2, and manifest.csv before rebuilding.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the written Combined manifest with the existing manifest loader.",
    )
    return parser.parse_args()


def parse_root_name(path: Path) -> RootSpec:
    match = ROOT_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unrecognized SEN12MS root directory name: {path}")
    roi_prefix, season, modality = match.groups()
    return RootSpec(
        roi_prefix=roi_prefix,
        season=season,
        modality=modality,
        path=path,
    )


def parse_tile_path(path: Path) -> TileRecord:
    if path.suffix.lower() not in RASTER_SUFFIXES:
        raise ValueError(f"Expected a TIFF file, got: {path}")

    match = FILE_NAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"Failed to parse raw SEN12MS TIFF filename: {path}. "
            "Expected '<ROI>_<season>_<modality>_<inner_index>_<patch_id>.tif'"
        )

    roi_prefix, season, modality, inner_index, patch_id = match.groups()
    return TileRecord(
        key=SampleKey(
            roi_prefix=roi_prefix,
            season=season,
            inner_index=inner_index,
            patch_id=patch_id,
        ),
        modality=modality,
        path=path,
    )


def discover_root_pairs(raw_root: Path) -> list[tuple[RootSpec, RootSpec]]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw SEN12MS root not found: {raw_root}")

    roots_by_pair: dict[tuple[str, str], dict[str, RootSpec]] = {}
    for child in sorted(raw_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        match = ROOT_NAME_RE.fullmatch(child.name)
        if match is None:
            continue

        spec = parse_root_name(child)
        if spec.modality == "s2_cloudy":
            continue

        pair_modalities = roots_by_pair.setdefault(spec.pair_key, {})
        if spec.modality in pair_modalities:
            raise ValueError(
                f"Duplicate raw root for {spec.pair_key} modality '{spec.modality}': "
                f"{pair_modalities[spec.modality].path} and {spec.path}"
            )
        pair_modalities[spec.modality] = spec

    if not roots_by_pair:
        raise FileNotFoundError(
            f"No SEN12MS *_s1/*_s2 roots found under raw root: {raw_root}"
        )

    missing_root_pairs: list[str] = []
    paired_roots: list[tuple[RootSpec, RootSpec]] = []
    for pair_key in sorted(roots_by_pair):
        modalities = roots_by_pair[pair_key]
        missing = [name for name in EXPECTED_MODALITIES if name not in modalities]
        if missing:
            missing_root_pairs.append(
                f"{pair_key[0]}_{pair_key[1]} missing {', '.join(missing)}"
            )
            continue
        paired_roots.append((modalities["s1"], modalities["s2"]))

    if missing_root_pairs:
        preview = "; ".join(missing_root_pairs[:5])
        raise FileNotFoundError(
            "Raw SEN12MS modality roots are incomplete. "
            f"First issues: {preview}"
        )

    return paired_roots


def iter_rasters(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES:
            yield path


def discover_tiles(root_spec: RootSpec) -> dict[SampleKey, Path]:
    tiles: dict[SampleKey, Path] = {}
    for path in iter_rasters(root_spec.path):
        record = parse_tile_path(path)
        if record.modality == "s2_cloudy":
            raise ValueError(
                f"Cloudy optical tile was discovered in a clean-only scan: {path}"
            )
        if record.modality != root_spec.modality:
            raise ValueError(
                f"Filename modality '{record.modality}' does not match root "
                f"modality '{root_spec.modality}' for {path}"
            )
        if (
            record.key.roi_prefix != root_spec.roi_prefix
            or record.key.season != root_spec.season
        ):
            raise ValueError(
                f"Filename ROI/season does not match containing root for {path}"
            )
        if record.key in tiles:
            raise ValueError(
                f"Duplicate raw {root_spec.modality} tile for sample_id "
                f"'{record.key.sample_id}': {tiles[record.key]} and {path}"
            )
        tiles[record.key] = path
    return tiles


def collect_pairs(
    raw_root: Path,
) -> tuple[list[tuple[SampleKey, Path, Path]], int, int, list[SampleKey], list[SampleKey]]:
    sar_tiles: dict[SampleKey, Path] = {}
    optical_tiles: dict[SampleKey, Path] = {}

    for sar_root, optical_root in discover_root_pairs(raw_root):
        sar_records = discover_tiles(sar_root)
        optical_records = discover_tiles(optical_root)

        duplicate_sar = sorted(set(sar_tiles).intersection(sar_records))
        if duplicate_sar:
            raise ValueError(
                "Duplicate canonical sample_id(s) across SAR roots: "
                f"{[key.sample_id for key in duplicate_sar[:5]]}"
            )
        duplicate_optical = sorted(set(optical_tiles).intersection(optical_records))
        if duplicate_optical:
            raise ValueError(
                "Duplicate canonical sample_id(s) across optical roots: "
                f"{[key.sample_id for key in duplicate_optical[:5]]}"
            )

        sar_tiles.update(sar_records)
        optical_tiles.update(optical_records)

    sar_keys = set(sar_tiles)
    optical_keys = set(optical_tiles)
    unmatched_sar = sorted(sar_keys - optical_keys)
    unmatched_optical = sorted(optical_keys - sar_keys)

    if unmatched_sar or unmatched_optical:
        message_lines = ["SEN12MS pairing mismatch detected."]
        message_lines.append(
            f"  SAR files discovered: {len(sar_tiles)}"
        )
        message_lines.append(
            f"  optical files discovered: {len(optical_tiles)}"
        )
        message_lines.append(
            f"  unmatched SAR files: {len(unmatched_sar)}"
        )
        if unmatched_sar:
            preview = ", ".join(key.sample_id for key in unmatched_sar[:5])
            message_lines.append(f"    first SAR-only sample_ids: {preview}")
        message_lines.append(
            f"  unmatched optical files: {len(unmatched_optical)}"
        )
        if unmatched_optical:
            preview = ", ".join(key.sample_id for key in unmatched_optical[:5])
            message_lines.append(f"    first optical-only sample_ids: {preview}")
        raise ValueError("\n".join(message_lines))

    matched_keys = sorted(sar_keys)
    pairs = [
        (key, sar_tiles[key], optical_tiles[key])
        for key in matched_keys
    ]
    return pairs, len(sar_tiles), len(optical_tiles), unmatched_sar, unmatched_optical


def ensure_clean_output(
    combined_root: Path,
    *,
    overwrite_manifest: bool,
    clean_output: bool,
) -> None:
    manifest_path = combined_root / "manifest.csv"
    s1_dir = combined_root / "S1"
    s2_dir = combined_root / "S2"

    if clean_output:
        if manifest_path.exists():
            manifest_path.unlink()
        for path in (s1_dir, s2_dir):
            if path.exists():
                shutil.rmtree(path)

    if manifest_path.exists() and not overwrite_manifest:
        raise FileExistsError(
            f"Manifest already exists at {manifest_path}. "
            "Pass --overwrite-manifest or --clean-output."
        )

    for modality_dir in (s1_dir, s2_dir):
        if modality_dir.is_dir():
            existing_entries = list(modality_dir.iterdir())
            if existing_entries:
                raise FileExistsError(
                    f"Output modality directory is not empty: {modality_dir}. "
                    "Pass --clean-output to rebuild from scratch."
                )

    combined_root.mkdir(parents=True, exist_ok=True)
    s1_dir.mkdir(parents=True, exist_ok=True)
    s2_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest_path.unlink()


def materialize_asset(source: Path, destination: Path, *, link_mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if link_mode == "symlink":
        try:
            destination.symlink_to(source.resolve())
        except OSError as exc:
            raise OSError(
                f"Failed to create symlink '{destination}' -> '{source.resolve()}'. "
                "Use --link-mode copy if symlinks are unavailable in this environment."
            ) from exc
        return

    if link_mode == "copy":
        shutil.copy2(source, destination)
        return

    raise ValueError(f"Unsupported link mode: {link_mode!r}")


def write_manifest(
    combined_root: Path,
    pairs: list[tuple[SampleKey, Path, Path]],
    *,
    link_mode: str,
) -> Path:
    manifest_path = combined_root / "manifest.csv"
    s1_dir = combined_root / "S1"
    s2_dir = combined_root / "S2"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()

        for key, sar_path, optical_path in pairs:
            sample_id = key.sample_id
            combined_s1 = s1_dir / f"{sample_id}.tif"
            combined_s2 = s2_dir / f"{sample_id}.tif"

            materialize_asset(sar_path, combined_s1, link_mode=link_mode)
            materialize_asset(optical_path, combined_s2, link_mode=link_mode)

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "output_S1": f"S1/{sample_id}.tif",
                    "output_S2": f"S2/{sample_id}.tif",
                }
            )

    return manifest_path


def validate_combined_output(combined_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from data_loader.combined_manifest import (  # noqa: WPS433
        load_combined_manifest_samples,
        validate_combined_manifest_samples,
    )

    samples = load_combined_manifest_samples(combined_root)
    validate_combined_manifest_samples(samples)


def build_combined_dataset(
    *,
    raw_root: Path,
    combined_root: Path,
    link_mode: str = "symlink",
    overwrite_manifest: bool = False,
    clean_output: bool = False,
    validate: bool = False,
) -> BuildStats:
    pairs, sar_count, optical_count, unmatched_sar, unmatched_optical = collect_pairs(
        raw_root
    )

    if not pairs:
        raise ValueError(f"No matched SEN12MS SAR/optical pairs found under {raw_root}")

    ensure_clean_output(
        combined_root,
        overwrite_manifest=overwrite_manifest,
        clean_output=clean_output,
    )
    write_manifest(combined_root, pairs, link_mode=link_mode)

    if validate:
        validate_combined_output(combined_root)

    return BuildStats(
        sar_files_discovered=sar_count,
        optical_files_discovered=optical_count,
        matched_pairs_written=len(pairs),
        unmatched_sar_files=len(unmatched_sar),
        unmatched_optical_files=len(unmatched_optical),
        materialization_mode=link_mode,
        validation_ran=validate,
    )


def main() -> None:
    args = parse_args()
    stats = build_combined_dataset(
        raw_root=args.raw_root.resolve(),
        combined_root=args.combined_root.resolve(),
        link_mode=args.link_mode,
        overwrite_manifest=args.overwrite_manifest,
        clean_output=args.clean_output,
        validate=args.validate,
    )

    print(f"raw root: {args.raw_root.resolve()}")
    print(f"combined root: {args.combined_root.resolve()}")
    print(f"SAR files discovered: {stats.sar_files_discovered}")
    print(f"optical files discovered: {stats.optical_files_discovered}")
    print(f"matched pairs written: {stats.matched_pairs_written}")
    print(f"unmatched SAR files: {stats.unmatched_sar_files}")
    print(f"unmatched optical files: {stats.unmatched_optical_files}")
    print(f"default materialization mode used: {stats.materialization_mode}")
    print(f"validation ran: {stats.validation_ran}")


if __name__ == "__main__":
    main()
