# `splits.py`

## Purpose

Provides deterministic train/validation splitting helpers for sample IDs.

## Why This File Exists

Dataset objects consume ID lists. Splitting IDs before dataset construction is reproducible and keeps data selection separate from raster loading.

## Dependencies

Internal: none.

External: `random`, `numpy`.

## Classes

No classes are defined.

## Functions

### `make_split(ids, val_frac, seed)`

Purpose: deterministically shuffle sorted IDs and split into train/validation lists.

Inputs: `ids`, `val_frac` in `[0,1]`, and RNG `seed`.

Outputs: `(train_ids, val_ids)`.

Side effects: none.

Preconditions: `val_frac` is between 0 and 1 inclusive.

Postconditions:

- IDs are sorted before seeded shuffle for stable results independent of caller order.
- If `val_frac > 0` and IDs are non-empty, validation receives at least one sample.

Where it is used: example scripts.

Why it exists: reproducibility across machines and ID-list input order.

Edge cases:

- Empty input returns two empty lists.
- `val_frac=1.0` returns all IDs as validation and no training IDs.

### `make_stratified_split(ids, val_frac, seed, labels, bins=5)`

Purpose: split IDs while preserving strata defined by integer labels or quantile-binned float labels.

Inputs:

- `ids`: sample IDs.
- `val_frac`: validation fraction.
- `seed`: RNG seed.
- `labels`: mapping from ID to numeric stratum/target.
- `bins`: maximum quantile bins for floating labels.

Outputs: `(train_ids, val_ids)`.

Side effects: none.

Preconditions:

- Every ID has a label.
- `val_frac` in `[0,1]`.
- `bins >= 1`.

Postconditions:

- Each stratum is independently shuffled and split.
- Final train and validation outputs are shuffled again.

Where it is used: defined but not exported from package `__all__` and not used by traced training code.

Why it exists: enables stratification by class, region, flood fraction, or other per-ID labels.

Edge cases:

- Constant floating labels collapse to one bin.
- Tiny strata receive one validation item when `val_frac > 0`, which can overrepresent validation for rare strata.

## Data Contracts

Inputs and outputs are ID lists. No raster or tensor data is touched.

ENFORCED:

- Fraction and bin ranges.
- Complete label mapping for stratified split.

ASSUMED:

- Stable sorted ID order is desired before randomization.

## Tensor Shape Expectations

None.

## Metadata Structure

None.

## Valid Mask Handling

None.

## Transform / Augmentation Behavior

None.

## Design Decisions

- Uses Python's `random.Random(seed)` for deterministic shuffles.
- Float labels are stratified by quantiles instead of fixed-width bins.

## Edge Cases and Failure Modes

- `make_stratified_split` can create validation samples from every stratum, producing a larger validation set than `int(total * val_frac)`.
- Integer-like floats may be treated as categorical only if not detected as float type by the helper.

## Modification Risks

- Changing sort-before-shuffle changes all historical splits for the same seed.
- Exporting `make_stratified_split` would make it part of the public API.

## Example Usage

```python
train_ids, val_ids = make_split(ids, val_frac=0.2, seed=1337)
```

## Open Questions / Ambiguities

- The intended production use of `make_stratified_split` is unclear because it is not exported from `data_loader.__init__`.

