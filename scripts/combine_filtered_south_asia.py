from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


RASTER_SUFFIXES = {".tif", ".tiff"}
OUTPUT_MODALITIES = (
    "S1",
    "Label",
    "S1OtsuLabel",
    "S2",
    "JRCWater",
    "S2IndexLabel",
)


@dataclass(frozen=True)
class ModalitySpec:
    output_name: str
    source_dir: str
    source_suffix: str
    required: bool = False


@dataclass(frozen=True)
class SplitSpec:
    split_name: str
    label_mode: str
    modalities: tuple[ModalitySpec, ...]


@dataclass
class SampleRecord:
    sample_id: str
    base_id: str
    split_name: str
    label_mode: str
    sources: dict[str, Path] = field(default_factory=dict)


SPLIT_SPECS = (
    SplitSpec(
        split_name="HandLabeled",
        label_mode="strong",
        modalities=(
            ModalitySpec("S1", "S1Hand", "_S1Hand", required=True),
            ModalitySpec("Label", "LabelHand", "_LabelHand", required=True),
            ModalitySpec("S1OtsuLabel", "S1OtsuLabelHand", "_S1OtsuLabelHand"),
            ModalitySpec("S2", "S2Hand", "_S2Hand"),
            ModalitySpec("JRCWater", "JRCWaterHand", "_JRCWaterHand"),
        ),
    ),
    SplitSpec(
        split_name="WeaklyLabeled",
        label_mode="weak",
        modalities=(
            ModalitySpec("S1", "S1Weak", "_S1Weak", required=True),
            ModalitySpec("Label", "S1OtsuLabelWeak", "_S1OtsuLabelWeak", required=True),
            ModalitySpec("S2", "S2Weak", "_S2Weak"),
            ModalitySpec("S2IndexLabel", "S2IndexLabelWeak", "_S2IndexLabelWeak"),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine HandLabeled and WeaklyLabeled tiles from "
            "datasets/FilteredSouthAsia into a single flat dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/FilteredSouthAsia"),
        help="Root directory containing HandLabeled and WeaklyLabeled.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory for the combined dataset. Defaults to <dataset-root>/Combined.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="How to materialize files in the combined dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace conflicting files already present in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned work without writing files.",
    )
    return parser.parse_args()


def list_rasters(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES
    )


def strip_expected_suffix(stem: str, suffix: str, path: Path) -> str:
    if not stem.endswith(suffix):
        raise ValueError(
            f"Unexpected file naming in {path}: expected stem to end with '{suffix}', got '{stem}'"
        )
    return stem[: -len(suffix)]


def collect_samples(dataset_root: Path, spec: SplitSpec) -> list[SampleRecord]:
    split_root = dataset_root / spec.split_name
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    anchor_spec = next(modality for modality in spec.modalities if modality.output_name == "S1")
    anchor_dir = split_root / anchor_spec.source_dir
    if not anchor_dir.is_dir():
        raise FileNotFoundError(f"Missing required modality directory: {anchor_dir}")

    records_by_base_id: dict[str, SampleRecord] = {}
    for path in list_rasters(anchor_dir):
        base_id = strip_expected_suffix(path.stem, anchor_spec.source_suffix, path)
        sample_id = path.stem
        records_by_base_id[base_id] = SampleRecord(
            sample_id=sample_id,
            base_id=base_id,
            split_name=spec.split_name,
            label_mode=spec.label_mode,
            sources={"S1": path},
        )

    for modality in spec.modalities:
        if modality.output_name == "S1":
            continue

        source_dir = split_root / modality.source_dir
        if not source_dir.exists():
            if modality.required:
                raise FileNotFoundError(f"Missing required modality directory: {source_dir}")
            continue

        for path in list_rasters(source_dir):
            base_id = strip_expected_suffix(path.stem, modality.source_suffix, path)
            if base_id not in records_by_base_id:
                raise ValueError(
                    f"{path} does not have a matching S1 tile in {anchor_dir}"
                )
            record = records_by_base_id[base_id]
            if modality.output_name in record.sources:
                raise ValueError(
                    f"Duplicate {modality.output_name} file for sample '{record.sample_id}'"
                )
            record.sources[modality.output_name] = path

    missing_required: list[str] = []
    for record in records_by_base_id.values():
        for modality in spec.modalities:
            if modality.required and modality.output_name not in record.sources:
                missing_required.append(f"{record.sample_id}:{modality.output_name}")
    if missing_required:
        preview = ", ".join(sorted(missing_required)[:5])
        raise ValueError(f"Missing required modality files: {preview}")

    return sorted(records_by_base_id.values(), key=lambda record: record.sample_id)


def materialize_file(source: Path, destination: Path, link_mode: str, overwrite: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if overwrite:
            destination.unlink()
        elif os.path.samefile(source, destination):
            return "reused"
        else:
            raise FileExistsError(
                f"Destination already exists and differs from source: {destination}"
            )

    if link_mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if link_mode == "hardlink":
                raise

    shutil.copy2(source, destination)
    return "copy"


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_manifest(
    output_root: Path,
    dataset_root: Path,
    rows: list[dict[str, str]],
) -> Path:
    manifest_path = output_root / "manifest.csv"
    fieldnames = [
        "sample_id",
        "base_id",
        "split_name",
        "label_mode",
    ]
    for modality in OUTPUT_MODALITIES:
        fieldnames.append(f"source_{modality}")
        fieldnames.append(f"output_{modality}")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path = output_root / "README.txt"
    summary_path.write_text(
        (
            "Combined FilteredSouthAsia dataset\n"
            f"Source root: {dataset_root}\n"
            f"Samples: {len(rows)}\n"
            "Canonical modality folders: S1, Label, S1OtsuLabel, S2, JRCWater, S2IndexLabel\n"
            "Files in each modality folder are renamed to the S1 sample ID so paired files share a common stem.\n"
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else (dataset_root / "Combined").resolve()

    all_records: list[SampleRecord] = []
    split_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()

    for spec in SPLIT_SPECS:
        records = collect_samples(dataset_root, spec)
        all_records.extend(records)
        split_counts[spec.split_name] = len(records)
        for record in records:
            for modality in record.sources:
                modality_counts[modality] += 1

    manifest_rows: list[dict[str, str]] = []
    write_counts: Counter[str] = Counter()
    copy_mode_counts: Counter[str] = Counter()

    for record in all_records:
        row: dict[str, str] = {
            "sample_id": record.sample_id,
            "base_id": record.base_id,
            "split_name": record.split_name,
            "label_mode": record.label_mode,
        }

        for modality in OUTPUT_MODALITIES:
            row[f"source_{modality}"] = ""
            row[f"output_{modality}"] = ""

        for modality, source_path in record.sources.items():
            destination = output_root / modality / f"{record.sample_id}{source_path.suffix.lower()}"
            row[f"source_{modality}"] = relative_to(source_path, dataset_root)
            row[f"output_{modality}"] = relative_to(destination, output_root)

            if not args.dry_run:
                materialize_result = materialize_file(
                    source=source_path,
                    destination=destination,
                    link_mode=args.link_mode,
                    overwrite=args.overwrite,
                )
                copy_mode_counts[materialize_result] += 1
            write_counts[modality] += 1

        manifest_rows.append(row)

    if args.dry_run:
        print("Dry run only. No files were written.")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = write_manifest(output_root, dataset_root, manifest_rows)
        print(f"Wrote manifest to {manifest_path}")

    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")
    print(f"Samples total: {len(all_records)}")
    for split_name in sorted(split_counts):
        print(f"  {split_name}: {split_counts[split_name]}")

    print("Modalities discovered:")
    for modality in OUTPUT_MODALITIES:
        count = modality_counts.get(modality, 0)
        if count:
            print(f"  {modality}: {count}")

    if not args.dry_run:
        print("Materialization mode counts:")
        for mode_name, count in sorted(copy_mode_counts.items()):
            print(f"  {mode_name}: {count}")


if __name__ == "__main__":
    main()
