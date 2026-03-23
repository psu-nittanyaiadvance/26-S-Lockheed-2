from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_COUNTRIES = ["India", "Pakistan", "Sri-Lanka"]
DEFAULT_SPLITS = ["HandLabeled", "WeaklyLabeled"]


def normalize_country(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def build_country_set(countries: Iterable[str]) -> set[str]:
    return {normalize_country(country) for country in countries}


def country_matches(filename: str, country_set: set[str]) -> bool:
    if "_" not in filename:
        return False
    prefix = filename.split("_", 1)[0]
    return normalize_country(prefix) in country_set


def copy_filtered(
    datasets_dir: Path,
    output_dir: Path,
    country_set: set[str],
    splits: Iterable[str],
    *,
    dry_run: bool,
    overwrite: bool,
) -> None:
    total = {"scanned": 0, "matched": 0, "copied": 0, "skipped": 0}

    for split in splits:
        src_root = datasets_dir / split
        if not src_root.exists():
            print(f"Skip missing split: {src_root}")
            continue

        stats = {"scanned": 0, "matched": 0, "copied": 0, "skipped": 0}

        for path in src_root.rglob("*"):
            if not path.is_file():
                continue
            stats["scanned"] += 1
            total["scanned"] += 1

            if not country_matches(path.name, country_set):
                continue

            stats["matched"] += 1
            total["matched"] += 1

            rel_path = path.relative_to(src_root)
            dest_path = output_dir / split / rel_path

            if dest_path.exists() and not overwrite:
                stats["skipped"] += 1
                total["skipped"] += 1
                continue

            if not dry_run:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest_path)

            stats["copied"] += 1
            total["copied"] += 1

        print(
            f"{split}: scanned={stats['scanned']} "
            f"matched={stats['matched']} copied={stats['copied']} "
            f"skipped={stats['skipped']}"
        )

    print(
        f"Total: scanned={total['scanned']} matched={total['matched']} "
        f"copied={total['copied']} skipped={total['skipped']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter datasets to India, Pakistan, and Sri Lanka while keeping the "
            "HandLabeled/WeaklyLabeled folder structure."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    datasets_dir = repo_root / "datasets"
    default_output = datasets_dir / "FilteredSouthAsia"

    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=datasets_dir,
        help="Path to the datasets directory (default: <repo>/datasets).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Output directory for filtered data.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=DEFAULT_COUNTRIES,
        help="Country names to include (matched against filename prefix).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Dataset split folders to include.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    country_set = build_country_set(args.countries)

    copy_filtered(
        args.datasets_dir,
        args.output_dir,
        country_set,
        args.splits,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
