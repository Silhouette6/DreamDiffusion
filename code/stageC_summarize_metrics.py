"""
Stage C - Summarize metrics_summary.json files across multiple experiment runs.

Input:  a parent folder containing multiple metric_summary.json files
        (each produced by stageC_turn_imgs_to_metrics.py)

        Example: python code/stageC_summarize_metrics.py --folder <包含metrics_summary.json的目录>
        
Output: prints overall averages and warnings to console
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Exclusion list – manually remove samples with no statistical value
# ---------------------------------------------------------------------------
EXCLUDED_SAMPLES = [
    # 'n04069434_9580',
]


def parse_args():
    parser = argparse.ArgumentParser("Stage C: Summarize metric_summary files")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder that contains one or more metric_summary.json files")
    return parser.parse_args()


def _collect_summaries(root: Path):
    """Find all metric_summary.json files under *root*."""
    summaries = {}
    # direct child
    if (root / "metrics_summary.json").exists():
        summaries[root.name] = root / "metrics_summary.json"
    # nested sub-folders
    for p in sorted(root.rglob("metrics_summary.json")):
        summaries[p.parent.name] = p
    return summaries


def main():
    args = parse_args()
    root = Path(args.folder)
    assert root.is_dir(), f"Folder not found: {root}"

    summaries = _collect_summaries(root)
    assert len(summaries) > 0, f"No metrics_summary.json found under {root}"
    print(f"Found {len(summaries)} metric_summary file(s)\n")

    # ---- gather per-sample metrics across all summaries ----
    all_samples = {}  # sample_name -> metrics dict
    summary_names = []

    for name, path in summaries.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary_names.append(name)
        per_sample = data.get("per_sample", {})
        for sample_name, metrics in per_sample.items():
            if sample_name in EXCLUDED_SAMPLES:
                continue
            all_samples[sample_name] = metrics

    print(f"Total samples (after exclusion): {len(all_samples)}")
    if EXCLUDED_SAMPLES:
        print(f"Excluded samples: {EXCLUDED_SAMPLES}")
    print()

    if len(all_samples) == 0:
        print("No samples to aggregate.")
        return

    # ---- warn about top1_accuracy_40way == 0 ----
    zero_acc_samples = [
        name for name, m in all_samples.items()
        if m.get("top1_accuracy_40way", -1) == 0
    ]
    print("=" * 60)
    print(f"top1_accuracy_40way == 0 samples ({len(zero_acc_samples)}):")
    print(f"  {repr(zero_acc_samples)}")
    print("  ^ copy to EXCLUDED_SAMPLES if needed")
    print("=" * 60)
    print()

    # ---- compute overall averages ----
    keys = ["inception_feature_distance", "top1_accuracy_40way", "clip_score_text"]
    overall = {}
    for k in keys:
        vals = [m[k] for m in all_samples.values() if k in m]
        if vals:
            overall[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "count": len(vals),
            }

    print("=" * 60)
    print(f"Overall aggregate ({len(all_samples)} samples):")
    print("=" * 60)
    for k, stats in overall.items():
        print(f"  {k}:")
        print(f"    mean  = {stats['mean']:.6f}")
        print(f"    std   = {stats['std']:.6f}")
        print(f"    min   = {stats['min']:.6f}")
        print(f"    max   = {stats['max']:.6f}")
        print(f"    count = {stats['count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
