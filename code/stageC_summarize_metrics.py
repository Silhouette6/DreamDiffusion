"""
Stage C - Summarize metrics_summary.json files across multiple experiment runs.

Input:  a parent folder containing multiple metric_summary.json files
        (each produced by stageC_turn_imgs_to_metrics.py)

        Example: python code/stageC_summarize_metrics.py --folder <包含metrics_summary.json的目录>

        Paired comparison:
          python code/stageC_summarize_metrics.py --folder results/generated/exp1 --baseline results/generated/baseline

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


METRIC_KEYS = ["inception_feature_distance", "top1_accuracy_40way", "clip_score_text"]

METRIC_DIRECTION = {
    "inception_feature_distance": "lower_better",
    "top1_accuracy_40way": "higher_better",
    "clip_score_text": "higher_better",
}


def parse_args():
    parser = argparse.ArgumentParser("Stage C: Summarize metric_summary files")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder that contains one or more metric_summary.json files")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Optional baseline folder for paired statistical comparison")
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


def _load_per_sample(root: Path):
    """Load and merge per-sample metrics from all summaries under *root*."""
    summaries = _collect_summaries(root)
    assert len(summaries) > 0, f"No metrics_summary.json found under {root}"
    print(f"Found {len(summaries)} metric_summary file(s)")

    all_samples = {}
    for name, path in summaries.items():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        per_sample = data.get("per_sample", {})
        for sample_name, metrics in per_sample.items():
            if sample_name in EXCLUDED_SAMPLES:
                continue
            all_samples[sample_name] = metrics
    return all_samples


def _print_aggregate(all_samples, label="Overall"):
    """Print aggregate statistics."""
    overall = {}
    for k in METRIC_KEYS:
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
    print(f"{label} aggregate ({len(all_samples)} samples):")
    print("=" * 60)
    for k, stats in overall.items():
        print(f"  {k}:")
        print(f"    mean  = {stats['mean']:.6f}")
        print(f"    std   = {stats['std']:.6f}")
        print(f"    min   = {stats['min']:.6f}")
        print(f"    max   = {stats['max']:.6f}")
        print(f"    count = {stats['count']}")
    print("=" * 60)
    return overall


def _paired_comparison(exp_samples, base_samples):
    """Paired statistical comparison between experiment and baseline.

    Uses paired t-test and Wilcoxon signed-rank test on the shared samples.
    """
    from scipy import stats as sp_stats

    shared = sorted(set(exp_samples) & set(base_samples))
    if len(shared) == 0:
        print("No shared samples between experiment and baseline.")
        return

    print()
    print("=" * 60)
    print(f"Paired comparison ({len(shared)} shared samples)")
    print("=" * 60)

    for k in METRIC_KEYS:
        exp_vals = np.array([exp_samples[s][k] for s in shared if k in exp_samples[s]])
        base_vals = np.array([base_samples[s][k] for s in shared if k in base_samples[s]])
        if len(exp_vals) != len(base_vals) or len(exp_vals) == 0:
            continue

        diff = exp_vals - base_vals
        direction = METRIC_DIRECTION.get(k, "unknown")

        if direction == "lower_better":
            improved = diff < 0
            arrow = "↓"
        else:
            improved = diff > 0
            arrow = "↑"

        mean_diff = float(np.mean(diff))
        pct_improved = float(improved.mean()) * 100

        t_stat, t_pval = sp_stats.ttest_rel(exp_vals, base_vals)

        try:
            w_stat, w_pval = sp_stats.wilcoxon(diff)
        except ValueError:
            w_stat, w_pval = float("nan"), float("nan")

        sig_t = "***" if t_pval < 0.001 else ("**" if t_pval < 0.01 else ("*" if t_pval < 0.05 else ""))
        sig_w = "***" if w_pval < 0.001 else ("**" if w_pval < 0.01 else ("*" if w_pval < 0.05 else ""))

        print(f"\n  {k} ({arrow} is better):")
        print(f"    baseline mean = {float(np.mean(base_vals)):.6f}")
        print(f"    experiment    = {float(np.mean(exp_vals)):.6f}")
        print(f"    mean diff     = {mean_diff:+.6f}")
        print(f"    % improved    = {pct_improved:.1f}%")
        print(f"    paired t-test : t={t_stat:.4f}, p={t_pval:.4f} {sig_t}")
        print(f"    Wilcoxon      : W={w_stat:.1f}, p={w_pval:.4f} {sig_w}")

    print()
    print("  Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print("=" * 60)


def main():
    args = parse_args()
    root = Path(args.folder)
    assert root.is_dir(), f"Folder not found: {root}"

    # ---- load experiment metrics ----
    all_samples = _load_per_sample(root)

    print(f"\nTotal samples (after exclusion): {len(all_samples)}")
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

    _print_aggregate(all_samples, label="Overall")

    # ---- paired comparison with baseline ----
    if args.baseline is not None:
        baseline_root = Path(args.baseline)
        assert baseline_root.is_dir(), f"Baseline folder not found: {baseline_root}"
        print(f"\nLoading baseline from: {baseline_root}")
        base_samples = _load_per_sample(baseline_root)
        _paired_comparison(all_samples, base_samples)


if __name__ == "__main__":
    main()
