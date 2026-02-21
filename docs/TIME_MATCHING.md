# Time-Matched SAR+Optical Stacks

This repo uses GEE to export 8-band stacks that align Sentinel-1 SAR (VV, VH) with
Sentinel-2 optical bands for the same spatial extent and time window.

**What `scripts/gee_download_time_matched.py` Produces**
- GeoTIFF stacks: `data/derived/gee_time_matched/<split>/<sample_id>.tif`
- Manifest CSV: `data/derived/gee_time_matched_manifest.csv`

The manifest columns (written by the downloader):
- `sample_id`
- `path`
- `status` (`ok`, `missing_s1`, `missing_s2`, `missing_both`, `error`)
- `s1_date_used`, `s2_date_used`
- `window_s1`, `window_s2`

**Expected Band Order**
The downloader constructs stacks as:
`[VV, VH, B2, B3, B4, B8, B11, B12]`

The S2 band list is configured in `configs/gee_time_match.yaml` (`bands_s2`).

**Run the Downloader**

```powershell
python scripts/gee_download_time_matched.py --config configs/gee_time_match.yaml
```

**Validation Steps**

```powershell
python scripts/smoke_test_time_matched.py
python scripts/smoke_test_loader.py
```

Programmatic checks:

```python
from src.data_loader.validate_dataset import validate_manifest_consistency

validate_manifest_consistency("data/derived/gee_time_matched_manifest.csv")
```

**Missing-Data Policies**
- `zeros`: returns a zero 8-band stack when missing (warning once).
- `skip`: removes missing samples at init time.
- `raise`: raises on access (debug only).

Use `zeros` or `skip` only with explicit reporting of missing percentages.
