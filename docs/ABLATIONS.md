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

**Planned Ablation Matrix (Modality + Weak-Label Viability)**

| Run | Modality | Pretrain | Train/Fine-tune | Key Purpose |
| --- | --- | --- | --- | --- |
| R1 | SAR | None | Strong only | Baseline |
| R2 | SAR | None | Weak only | Weak-label viability |
| R3 | SAR | None | Strong + Weak | Mixture of labels viability |
| R4 | SAR | Weak Sen1Floods11 | Strong finetune | Weak-label benefit (SAR) |
| R5 | SAR | SSL SEN12MS | Strong finetune | Modality baseline |
| R6 | SAR+Opt | SSL SEN12MS | Strong finetune | Modality effect (SSL-controlled) |

**Handling Missing Time-Matched**
- `zeros` is allowed only if you report the % missing.
- `skip` changes dataset distribution: requires reporting and is only comparable if both arms use `skip`.
- `raise` is for debugging only; do not use for final runs.
