# `discover_ids.py`

## Purpose

Discovers raster IDs from directories and computes image/mask intersections.

## Why This File Exists

Many dataset constructors accept ID lists. This module provides small utilities for building those lists from file trees without loading rasters.

## Dependencies

Internal: none.

External: `pathlib.Path`, `warnings`.

## Classes

No classes are defined.

## Functions

### `list_ids_from_dir(img_root, recursive=False)`

Purpose: return sorted file stems for `.tif` and `.tiff` files under a directory.

Inputs:

- `img_root`: directory to scan.
- `recursive`: whether to use recursive globbing.

Outputs: sorted list of unique stems.

Side effects: emits warnings when multiple files share the same stem.

Preconditions: `img_root` exists and is a directory.

Postconditions: returned IDs are unique because stems are dictionary keys.

Where it is used: intended for dataset preparation and pairing workflows.

Why it exists: keeps ID discovery consistent across SAR, optical, and mask directories.

Edge cases:

- Duplicate stems under different directories collapse to one ID with a warning.
- Non-TIFF files are ignored.

### `paired_ids(img_root, mask_root, recursive_images=False, recursive_masks=False)`

Purpose: compute the intersection between image IDs and mask IDs and return pairing stats.

Inputs:

- `img_root`: image directory.
- `mask_root`: mask directory.
- `recursive_images`: recursive image scan flag.
- `recursive_masks`: recursive mask scan flag.

Outputs: `(paired, report)` where `paired` is sorted list of IDs and `report` contains counts.

Side effects: calls `list_ids_from_dir`, so duplicate warnings may be emitted.

Preconditions: both roots exist and are directories.

Postconditions: `paired` contains IDs present in both directories.

Where it is used: general utility; not directly imported by current training script.

Why it exists: allows users to avoid passing IDs that have no label in supervised modes.

Edge cases: it reports counts only, not the missing ID names.

## Data Contracts

ID equals TIFF file stem. Directory membership, not raster contents, defines discovery.

ENFORCED:

- Roots must be directories.

ASSUMED:

- File stem is the stable sample ID.
- `.tif` and `.tiff` extensions are the only relevant image formats.

ACCIDENTAL:

- Recursive duplicate stems are collapsed and only warned, which may hide separate acquisitions with the same stem.

## Tensor Shape Expectations

None.

## Metadata Structure

None.

## Valid Mask Handling

None.

## Transform / Augmentation Behavior

None.

## Design Decisions

- Intersection is used for supervised pairing rather than image-only or mask-only union.
- Warning rather than failure on duplicate stems makes the helper permissive for exploratory use.

## Edge Cases and Failure Modes

- Empty directories return empty lists rather than raising.
- Duplicate stems can lead to a discovered ID that maps ambiguously when constructing datasets.

## Modification Risks

- Returning paths instead of stems would break callers that pass IDs to datasets.
- Treating duplicates as hard errors would improve safety but may disrupt exploratory scripts.

## Example Usage

```python
ids, report = paired_ids("S1Hand", "LabelHand")
print(report)
```

## Open Questions / Ambiguities

- Whether duplicate stems should be fatal in production workflows is not specified.

