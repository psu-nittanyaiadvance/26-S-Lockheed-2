# Dataset Layout

This repo expects a Sen1Floods11-derived, region-filtered dataset with weak and strong labels.
Paths below are relative to the repo root. Replace any `<...>` placeholders with your local values.

**Primary Roots**
- Images (weak): `datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak`
- Masks (weak): `datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak`
- Images (strong): `datasets/FilteredSouthAsia/HandLabeled/S1Hand`
- Masks (strong): `datasets/FilteredSouthAsia/HandLabeled/LabelHand`

**Derived Artifacts**
- Time manifest (input to GEE): `data/derived/sen1floods11_time_manifest.csv`
- Time-matched stacks: `data/derived/gee_time_matched/<split>/<sample_id>.tif`
- Time-matched manifest: `data/derived/gee_time_matched_manifest.csv`

**ID Convention**
- Sample IDs are the filename stems of SAR GeoTIFFs (e.g., `tile_000123` from `tile_000123.tif`).
- When you pass IDs to `SARDataset`, the loader resolves `img_root/<id>.tif` (or `.tiff`).
- When you pass full paths, the loader uses them directly and infers the ID from the stem.

**Quick ID Discovery**
Use the helper in `src/data_loader/discover_ids.py` to avoid inconsistent lists:

```python
from src.data_loader.discover_ids import paired_ids

ids, report = paired_ids(
    "datasets/FilteredSouthAsia/WeaklyLabeled/S1Weak",
    "datasets/FilteredSouthAsia/WeaklyLabeled/S1OtsuLabelWeak",
)
print(report)
```

**Common Layout Pitfalls**
- Wrong roots: If you point `img_root` at masks (or vice versa), shapes and channel counts will be wrong.
- Mixed labeled/unlabeled: Use consistent `mode` and mask roots; the collate step rejects mixed batches.
- Missing time-matched manifest: `use_time_matched=True` will fall back to filesystem checks, but you should
  generate `data/derived/gee_time_matched_manifest.csv` for reproducibility.
