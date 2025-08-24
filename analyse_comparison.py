# analyze_comparisons.py — summarize baseline vs augmented results

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel



def analyze(input_csv: str, out_dir: str, save_plot: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Required columns check
    required = {
        "id", "subject", "question",
        "baseline_correctness", "baseline_completeness", "baseline_clarity",
        "baseline_conciseness", "baseline_overall",
        "aug_correctness", "aug_completeness", "aug_clarity",
        "aug_conciseness", "aug_overall",
        "delta_correctness", "delta_completeness", "delta_clarity",
        "delta_conciseness", "delta_overall"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    metrics = ["correctness", "completeness", "clarity", "conciseness", "overall"]

    # Overall aggregates
    summary = {}
    for m in metrics:
        base_mean = float(df[f"baseline_{m}"].mean())
        aug_mean = float(df[f"aug_{m}"].mean())
        delta_mean = float(df[f"delta_{m}"].mean())
        base_std = float(df[f"baseline_{m}"].std(ddof=0))
        aug_std = float(df[f"aug_{m}"].std(ddof=0))
        win_rate = float((df[f"delta_{m}"] > 0).mean())  # fraction where aug > base

        summary[m] = {
            "baseline_mean": round(base_mean, 3),
            "baseline_std": round(base_std, 3),
            "augmented_mean": round(aug_mean, 3),
            "augmented_std": round(aug_std, 3),
            "delta_mean": round(delta_mean, 3),
            "win_rate": round(win_rate, 3)  # proportion of rows with positive gain
        }
        # Paired t-test for overall score
    t_stat, p_val = ttest_rel(df["aug_overall"], df["baseline_overall"])
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    # Per-subject summary
    per_subject = (
        df.groupby("subject", dropna=False)
          .agg({
              "baseline_overall": "mean",
              "aug_overall": "mean",
              "delta_overall": ["mean", "count"]
          })
    )
    per_subject.columns = ["baseline_overall_mean", "aug_overall_mean", "delta_overall_mean", "n_rows"]
    per_subject = per_subject.reset_index().sort_values("delta_overall_mean", ascending=False)
    per_subject_path = out / "per_subject_summary.csv"
    per_subject.to_csv(per_subject_path, index=False)

    # Per-question summary (one row per id; take mean deltas if multiple rows per id)
    per_q = (
        df.groupby(["id", "subject", "question"], dropna=False)[[f"delta_{m}" for m in metrics]]
          .mean()
          .reset_index()
          .sort_values("delta_overall", ascending=False)
    )
    per_q_path = out / "per_question_deltas.csv"
    per_q.to_csv(per_q_path, index=False)

    # Save metrics JSON
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional histogram plot of overall deltas
    plot_path = out / "delta_overall_hist.png"
    if save_plot:
        plt.figure()
        df["delta_overall"].hist(bins=20)
        plt.title("Distribution of Overall Improvement (Aug - Base)")
        plt.xlabel("Delta Overall")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    print(f"✅ Saved:\n- {metrics_path}\n- {per_subject_path}\n- {per_q_path}")
    if save_plot:
        print(f"- {plot_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze comparison results from compare_solutions.csv")
    parser.add_argument("--input", default="runs/compare_solutions.csv", help="Path to compare_solutions.csv")
    parser.add_argument("--out", default="runs/analysis", help="Directory to write summaries/plots")
    parser.add_argument("--no-plot", action="store_true", help="Disable histogram plot")
    args = parser.parse_args()

    analyze(args.input, args.out, save_plot=not args.no_plot)

