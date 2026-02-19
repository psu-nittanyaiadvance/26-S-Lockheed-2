from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union
import warnings


def list_ids_from_dir(img_root: Union[str, Path], recursive: bool = False) -> List[str]:
    """
    Return sorted stems for *.tif/*.tiff in a directory.

    If recursive=True, search nested directories. Duplicate stems under root emit a warning.
    """
    root = Path(img_root)
    if not root.is_dir():
        raise ValueError(f"img_root is not a directory: {root}")
    if recursive:
        paths = list(root.rglob("*.tif")) + list(root.rglob("*.tiff"))
    else:
        paths = list(root.glob("*.tif")) + list(root.glob("*.tiff"))
    paths = sorted(paths, key=lambda p: str(p))

    stems: Dict[str, List[Path]] = {}
    for p in paths:
        stems.setdefault(p.stem, []).append(p)

    for stem, stem_paths in stems.items():
        if len(stem_paths) > 1:
            max_items = 5
            shown = [str(p) for p in stem_paths[:max_items]]
            extra = len(stem_paths) - len(shown)
            suffix = f", ... (+{extra} more)" if extra > 0 else ""
            warnings.warn(
                "Duplicate stem detected "
                f"'{stem}' in {root}: {', '.join(shown)}{suffix}"
            )

    return sorted(stems.keys())


def paired_ids(
    img_root: Union[str, Path],
    mask_root: Union[str, Path],
    recursive_images: bool = False,
    recursive_masks: bool = False,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Return the intersection of image/mask IDs and a missing-count report.
    """
    img_ids = set(list_ids_from_dir(img_root, recursive=recursive_images))
    mask_ids = set(list_ids_from_dir(mask_root, recursive=recursive_masks))

    paired = sorted(img_ids.intersection(mask_ids))
    report = {
        "images_total": len(img_ids),
        "masks_total": len(mask_ids),
        "paired_total": len(paired),
        "missing_in_images": len(mask_ids - img_ids),
        "missing_in_masks": len(img_ids - mask_ids),
    }
    return paired, report
