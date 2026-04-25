# `filter_south_asia.py`

## Purpose

Copies files whose filename prefix matches selected countries into a filtered dataset directory while preserving split-relative folder structure.

## Why This File Exists

The repository appears to focus on a South Asia subset. This utility creates that subset from broader dataset folders based on country prefixes in filenames.

## Dependencies

Internal: none.

External:

- `argparse`
- `re`
- `shutil`
- `pathlib.Path`

## Classes

No classes are defined.

## Functions

### `normalize_country(name)`

Purpose: normalize a country string to lowercase alphanumeric characters.

Inputs: country name.

Outputs: normalized string.

### `build_country_set(countries)`

Purpose: normalize a collection of country names into a set.

Inputs: iterable of names.

Outputs: `set[str]`.

### `country_matches(filename, country_set)`

Purpose: check whether filename prefix before first underscore matches a normalized country.

Inputs: filename and normalized country set.

Outputs: bool.

Preconditions: filename convention uses `<country>_...`.

### `copy_filtered(datasets_dir, output_dir, country_set, splits, dry_run, overwrite)`

Purpose: scan split directories, copy matching files, and print copy statistics.

Inputs:

- source dataset directory;
- output directory;
- normalized country set;
- split names;
- dry-run and overwrite flags.

Outputs: none returned; prints stats.

Side effects:

- Recursively scans files.
- Creates directories and copies files with metadata when `dry_run=False`.

Preconditions:

- Split directories may exist under `datasets_dir`; missing splits are skipped.

Postconditions:

- Matching files are copied to `output_dir/<split>/<relative_path>` unless skipped.

### `parse_args()`

Purpose: define CLI arguments and defaults.

Outputs: `argparse.Namespace`.

### `main()`

Purpose: parse CLI args, build country set, and run copying.

## Data Contracts

This script works on files, not tensors. It assumes filenames begin with country names followed by `_`.

ENFORCED:

- Missing split directories are skipped with a message.

ASSUMED:

- Country identity is encoded only in filename prefix.
- Preserving relative paths under each split is enough to keep dataset structure.

ACCIDENTAL:

- Files without `_` never match, even if country is in another part of the path.

## Tensor Shape Expectations

None.

## Metadata Structure

None.

## Valid Mask Handling

None.

## Transform / Augmentation Behavior

None.

## Design Decisions

- Uses copy rather than symlink to create standalone filtered datasets.
- Defaults target `India`, `Pakistan`, and `Sri-Lanka`.
- Defaults scan `HandLabeled` and `WeaklyLabeled`.

## Edge Cases and Failure Modes

- Existing destination files are skipped unless `--overwrite` is set.
- Dry run increments `copied` count for files that would be copied.
- Country normalization removes punctuation, so `Sri-Lanka` and `Sri Lanka` match.

## Modification Risks

- Changing prefix parsing changes subset membership.
- Copying all file types can include non-raster sidecar files if their names match.

## Example Usage

```bash
python -m data_loader.filter_south_asia --dry-run
```

## Open Questions / Ambiguities

- Whether country membership should be derived from metadata rather than filename prefix.

