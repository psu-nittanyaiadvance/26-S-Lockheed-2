from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import OpticalDataset, SARDataset  # noqa: E402
from data_loader.combined_manifest import load_combined_manifest_samples  # noqa: E402


def _max_mean_diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a - b).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _profile(ds: Any) -> Dict[str, Any]:
    if hasattr(ds, "quantile_profile_summary"):
        return ds.quantile_profile_summary()
    return {}


def _compare_dataset(
    name: str,
    uncached: Any,
    cached: Any,
    indices: Iterable[int],
) -> Dict[str, Any]:
    index_list = list(indices)
    max_abs = 0.0
    mean_total = 0.0
    n = 0
    masks_equal = True

    t0 = time.perf_counter()
    refs = [uncached[i] for i in index_list]
    uncached_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = [cached[i] for i in index_list]
    fill_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = [cached[i] for i in index_list]
    hit_seconds = time.perf_counter() - t0

    for ref, hit in zip(refs, hits):
        ref_img, _ref_mask, ref_meta = ref
        hit_img, _hit_mask, hit_meta = hit
        item_max, item_mean = _max_mean_diff(ref_img, hit_img)
        max_abs = max(max_abs, item_max)
        mean_total += item_mean
        n += 1
        masks_equal = masks_equal and torch.equal(
            ref_meta["valid_mask"],
            hit_meta["valid_mask"],
        )

    return {
        "name": name,
        "samples": n,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_total / max(n, 1),
        "valid_masks_equal": masks_equal,
        "uncached_seconds": uncached_seconds,
        "cache_fill_seconds": fill_seconds,
        "cache_hit_seconds": hit_seconds,
        "uncached_profile": _profile(uncached),
        "cached_profile": _profile(cached),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact quantile cache equivalence.")
    parser.add_argument(
        "--combined-root",
        type=Path,
        default=ROOT / "datasets" / "FilteredSouthAsia" / "Combined",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache" / "quantiles")
    parser.add_argument("--sar-log-transform", action="store_true")
    parser.add_argument("--optical-clip-percentile", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    os.environ["LOCKDOCKS_PROFILE_QUANTILES"] = "1"

    samples = load_combined_manifest_samples(args.combined_root, require_label=False)
    selected = samples[: min(args.samples, len(samples))]
    indices = list(range(len(selected)))

    sar_paths = [sample.sar_path for sample in selected]
    optical_paths = [sample.optical_path for sample in selected]

    sar_kwargs = {
        "img_root": None,
        "mask_root": None,
        "ids_or_paths": sar_paths,
        "mode": "none",
        "ids_are_paths": True,
        "expected_img_bands": 2,
        "log_transform": args.sar_log_transform,
    }
    optical_kwargs = {
        "img_root": None,
        "mask_root": None,
        "ids_or_paths": optical_paths,
        "mode": "none",
        "ids_are_paths": True,
        "expected_img_bands": 13,
        "reflectance_clip_percentile": args.optical_clip_percentile,
    }

    reports = [
        _compare_dataset(
            "sar",
            SARDataset(**sar_kwargs, quantile_cache_root=None),
            SARDataset(**sar_kwargs, quantile_cache_root=args.cache_root),
            indices,
        ),
        _compare_dataset(
            "optical",
            OpticalDataset(**optical_kwargs, quantile_cache_root=None),
            OpticalDataset(**optical_kwargs, quantile_cache_root=args.cache_root),
            indices,
        ),
    ]

    failed = False
    for report in reports:
        print(f"\n[{report['name']}]")
        print(f"samples: {report['samples']}")
        print(f"max_abs_diff: {report['max_abs_diff']:.8g}")
        print(f"mean_abs_diff: {report['mean_abs_diff']:.8g}")
        print(f"valid_masks_equal: {report['valid_masks_equal']}")
        print(f"uncached_seconds: {report['uncached_seconds']:.4f}")
        print(f"cache_fill_seconds: {report['cache_fill_seconds']:.4f}")
        print(f"cache_hit_seconds: {report['cache_hit_seconds']:.4f}")
        print(f"uncached_profile: {report['uncached_profile']}")
        print(f"cached_profile: {report['cached_profile']}")
        if report["max_abs_diff"] > args.tolerance or not report["valid_masks_equal"]:
            failed = True

    if failed:
        raise SystemExit("Quantile cache verification failed.")


if __name__ == "__main__":
    main()
