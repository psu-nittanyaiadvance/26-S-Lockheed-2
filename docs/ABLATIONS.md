# Ablation Guardrails

This document defines ablation axes and rules to prevent confounds when reporting results.

**Axes**
- Modality: SAR-only vs SAR+Optical (time-matched).
- Labels: strong-only vs weak-only vs weak->strong (separate phase).
- Dataset domain: Sen1Floods11 region-filtered vs any external pretraining dataset.
- Time-matching availability: ok vs missing_s1 vs missing_s2 vs missing_both.

**No-Confound Rules**
- When comparing modality, keep label regime identical.
- When comparing weak-label usage, keep modality identical.
- Any pretraining MUST be treated as a separate axis; never compare pretrained baseline to non-pretrained multimodal.
- Splits must be identical across runs: store split IDs to disk and reuse.

**Required Reporting Checklist Per Run**
- Dataset roots (image + mask roots).
- Split seed + exact train/val IDs file paths.
- Mode (`weak`, `strong`, `none`).
- `use_time_matched` + missing policy (`zeros`, `skip`, `raise`).
- Normalization config and whether mean/std were computed on train split only.
- Channel counts (SAR-only=2; SAR+TM=8).

**Recommended Minimal Ablation Table**

| Modality | Labels | Notes |
| --- | --- | --- |
| SAR-only | strong-only | Baseline |
| SAR+TM | strong-only | Same IDs, same split |
| SAR-only | weak-only | Same IDs, same split |
| SAR+TM | weak-only | Same IDs, same split |

Optional later phase: weak->strong (treat as separate label-regime axis, not directly comparable).

**Handling Missing Time-Matched**
- `zeros` is allowed only if you report the % missing.
- `skip` changes dataset distribution: requires reporting and is only comparable if both arms use `skip`.
- `raise` is for debugging only; do not use for final runs.
