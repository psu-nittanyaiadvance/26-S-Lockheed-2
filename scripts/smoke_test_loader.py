from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader

from src.data_loader.collate import default_collate
from src.data_loader.discover_ids import paired_ids
from src.data_loader.sar_dataset import SARDataset
from src.data_loader.validate_dataset import validate_sample_shapes, validate_time_matched


def _build_dataset(
    name: str,
    img_root: Path,
    mask_root: Path,
    mode: str,
    use_time_matched: bool,
    time_matched_root: Path,
    time_matched_manifest: Path,
    missing_policy: str,
) -> Optional[SARDataset]:
    if not img_root.exists():
        print(f"{name}: missing img_root={img_root}")
        return None
    if mode != "none" and not mask_root.exists():
        print(f"{name}: missing mask_root={mask_root}")
        return None

    ids, report = paired_ids(img_root, mask_root) if mode != "none" else ([], {})
    if mode == "none":
        ids = sorted(p.stem for p in img_root.glob("*.tif"))[:32]
    else:
        ids = ids[:32]

    if not ids:
        print(f"{name}: no paired samples found (report={report})")
        return None

    ds = SARDataset(
        img_root=img_root,
        mask_root=mask_root,
        ids_or_paths=ids,
        mode=mode,
        use_time_matched=use_time_matched,
        time_matched_root=time_matched_root,
        time_matched_manifest=time_matched_manifest,
        time_matched_missing_policy=missing_policy,
    )
    return ds


def _pull_two_batches(ds: SARDataset) -> None:
    loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=False, num_workers=0, collate_fn=default_collate)
    it = iter(loader)
    for _ in range(2):
        try:
            next(it)
        except StopIteration:
            break


def _summary_line(name: str, ds: SARDataset) -> str:
    img, mask, meta = ds[0]
    return (
        f"{name}: samples={len(ds)} img_shape={tuple(img.shape)} "
        f"mask_present={mask is not None} time_matched={'time_matched' in meta} "
        f"missing_policy={ds.time_matched_missing_policy if ds.use_time_matched else 'none'}"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tm_root = repo_root / "data" / "derived" / "gee_time_matched"
    tm_manifest = repo_root / "data" / "derived" / "gee_time_matched_manifest.csv"
    missing_policy = "zeros"

    weak_img_root = repo_root / "datasets" / "FilteredSouthAsia" / "WeaklyLabeled" / "S1Weak"
    weak_mask_root = repo_root / "datasets" / "FilteredSouthAsia" / "WeaklyLabeled" / "S1OtsuLabelWeak"

    strong_img_root = repo_root / "datasets" / "FilteredSouthAsia" / "HandLabeled" / "S1Hand"
    strong_mask_root = repo_root / "datasets" / "FilteredSouthAsia" / "HandLabeled" / "LabelHand"

    datasets: Tuple[Tuple[str, Optional[SARDataset]], ...] = (
        (
            "weak",
            _build_dataset(
                "weak",
                weak_img_root,
                weak_mask_root,
                mode="weak",
                use_time_matched=True,
                time_matched_root=tm_root,
                time_matched_manifest=tm_manifest,
                missing_policy=missing_policy,
            ),
        ),
        (
            "strong",
            _build_dataset(
                "strong",
                strong_img_root,
                strong_mask_root,
                mode="strong",
                use_time_matched=True,
                time_matched_root=tm_root,
                time_matched_manifest=tm_manifest,
                missing_policy=missing_policy,
            ),
        ),
    )

    for name, ds in datasets:
        if ds is None:
            continue
        validate_sample_shapes(ds, n=16)
        validate_time_matched(ds, n=16)
        _pull_two_batches(ds)
        print(_summary_line(name, ds))


if __name__ == "__main__":
    main()
