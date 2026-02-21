from __future__ import annotations

from pathlib import Path

from src.data_loader.sar_dataset import SARDataset


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    img_root = repo_root / "datasets" / "FilteredSouthAsia" / "WeaklyLabeled" / "S1Weak"
    mask_root = repo_root / "datasets" / "FilteredSouthAsia" / "WeaklyLabeled" / "S1OtsuLabelWeak"

    if not img_root.exists():
        print(f"Missing image root: {img_root}")
        return

    ids = [p.stem for p in sorted(img_root.glob("*.tif"))[:5]]
    if not ids:
        print(f"No samples found under: {img_root}")
        return

    ds = SARDataset(
        img_root=img_root,
        mask_root=mask_root,
        ids_or_paths=ids,
        mode="weak",
        use_time_matched=True,
        time_matched_root=repo_root / "data" / "derived" / "gee_time_matched",
        time_matched_manifest=repo_root / "data" / "derived" / "gee_time_matched_manifest.csv",
        time_matched_missing_policy="zeros",
    )

    for i in range(len(ds)):
        img, mask, meta = ds[i]
        tm = meta.get("time_matched")
        tm_status = meta.get("time_matched_status")
        print(
            f"{meta['id']}: img={tuple(img.shape)} mask={tuple(mask.shape) if mask is not None else None} "
            f"tm={tuple(tm.shape) if tm is not None else None} status={tm_status}"
        )


if __name__ == "__main__":
    main()
