from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union


def list_ids_from_dir(img_root: Union[str, Path]) -> List[str]:
    """
    Return sorted stems for *.tif/*.tiff in a directory.
    """
    root = Path(img_root)
    if not root.is_dir():
        raise ValueError(f"img_root is not a directory: {root}")
    ids = {p.stem for p in root.glob("*.tif")}
    ids.update(p.stem for p in root.glob("*.tiff"))
    return sorted(ids)


def paired_ids(
    img_root: Union[str, Path], mask_root: Union[str, Path]
) -> Tuple[List[str], Dict[str, int]]:
    """
    Return the intersection of image/mask IDs and a missing-count report.
    """
    img_ids = set(list_ids_from_dir(img_root))
    mask_ids = set(list_ids_from_dir(mask_root))

    paired = sorted(img_ids.intersection(mask_ids))
    report = {
        "images_total": len(img_ids),
        "masks_total": len(mask_ids),
        "paired_total": len(paired),
        "missing_in_images": len(mask_ids - img_ids),
        "missing_in_masks": len(img_ids - mask_ids),
    }
    return paired, report
