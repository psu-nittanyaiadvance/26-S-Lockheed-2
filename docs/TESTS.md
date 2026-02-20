# Data Utility Smoke Tests

This document summarizes the data-loading smoke tests and explains why they matter
relative to the repo’s data and time-matching docs.

**What Passed**
All checks in `tests/test_data_utils.py` passed. The suite covers:
1. Recursive ID discovery finds nested SAR tiles and warns on duplicate stems.
2. The time-matched zeros warning is emitted exactly once and is validated by `validate_time_matched`.
3. `time_matched_missing_policy="skip"` removes missing stacks at init time.
4. `compute_running_mean_std` rejects datasets with transforms enabled when guardrails are on.
5. Band-count guards raise on mismatched SAR or time-matched stack channels.

**Why These Tests Are Good (Context From Docs)**
1. `docs/DATA.md` emphasizes stable ID discovery and avoiding inconsistent lists. The recursive
   discovery test ensures nested layouts are handled and duplicate stems are flagged.
2. `docs/TIME_MATCHING.md` and `docs/time_matched_gee.md` define the 8-band time-matched
   expectation and missing-data policies. The warning and skip-policy tests verify those rules
   are enforced in code and validated consistently.
3. `docs/DATALOADER.md` describes normalization and raw-stat computation. The guardrails test
   prevents accidental stats collection with transforms or pre-normalized data, reducing leakage.
4. `docs/ABLATIONS.md` requires consistent channel counts across runs. The band-count checks
   make it harder to silently mix 2-band SAR with 8-band time-matched stacks.
5. `docs/DATALOADER.md` notes that default collate forbids mixed labeled/unlabeled batches.

**How To Run**
From the repo root:

```powershell
python tests\test_data_utils.py
```
