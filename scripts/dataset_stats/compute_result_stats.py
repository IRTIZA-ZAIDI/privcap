"""Compute dataset and result statistics from PRIVCAP `*_all_tasks.json` files.

Each `*_all_tasks.json` (produced by the notebooks under
notebooks/evaluation/<dataset>/<model>/) has the shape:

    {
      "task3": {
        "temp=0.1": {
          "explicit": {"metrics": {"category_metrics": {...}, "macro_f1": ..., ...},
                        "n_samples": ..., "results": [{"id", "gt_combined",
                        "pred_labels", "all_runs", ...}, ...]},
          "no_leak": {...}
        },
        "temp=1.0": {...}
      },
      "task4": {...},
      "_meta": {"slug", "dataset", "n_samples", "temperatures", "num_runs", "seeds", ...}
    }

Ground truth (gt_combined) is fixed per sample and identical across models,
so category support counts pulled from any one model's file describe the
*dataset*, not that model's performance. This script reports both:

  1. dataset_label_distribution.csv  -- category support counts per task/condition,
     read from a single reference model (ground truth is model-independent).
  2. model_metrics.csv               -- macro F1/precision/recall per model, task,
     temperature, and condition, for cross-model comparison.

Usage:
    python compute_result_stats.py --results-dir /path/to/VISPR --out-dir ./out
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_all_tasks(model_dir: Path) -> dict | None:
    candidates = sorted(model_dir.glob("*_all_tasks.json"))
    if not candidates:
        return None
    with open(candidates[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def iter_task_blocks(data: dict):
    for task_name, temps in data.items():
        if task_name == "_meta" or not isinstance(temps, dict):
            continue
        for temp_label, conditions in temps.items():
            if not isinstance(conditions, dict):
                continue
            for condition, block in conditions.items():
                if isinstance(block, dict) and "metrics" in block:
                    yield task_name, temp_label, condition, block


def label_distribution(data: dict) -> pd.DataFrame:
    rows = []
    for task_name, temp_label, condition, block in iter_task_blocks(data):
        cat_metrics = block["metrics"].get("category_metrics", {})
        for category, m in cat_metrics.items():
            rows.append(
                {
                    "task": task_name,
                    "temperature": temp_label,
                    "condition": condition,
                    "category": category,
                    "support": m.get("support"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # support is identical across temperatures for the same task/condition
    # (ground truth doesn't change with decoding temperature) -- collapse it.
    return (
        df.drop_duplicates(subset=["task", "condition", "category"])
        .drop(columns="temperature")
        .sort_values(["task", "condition", "support"], ascending=[True, True, False])
    )


def model_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        data = load_all_tasks(model_dir)
        if data is None:
            continue
        for task_name, temp_label, condition, block in iter_task_blocks(data):
            m = block["metrics"]
            rows.append(
                {
                    "model": model_dir.name,
                    "task": task_name,
                    "temperature": temp_label,
                    "condition": condition,
                    "n_samples": block.get("n_samples"),
                    "macro_f1": m.get("macro_f1"),
                    "macro_precision": m.get("macro_precision"),
                    "macro_recall": m.get("macro_recall"),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True, type=Path, help="Dataset results dir, e.g. .../VISPR (contains one subdir per model)")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory to write CSV output")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    first_model_dir = next((p for p in sorted(args.results_dir.iterdir()) if p.is_dir()), None)
    if first_model_dir is None:
        print(f"No model subdirectories found under {args.results_dir}")
        return
    ref_data = load_all_tasks(first_model_dir)
    if ref_data is not None:
        dist_df = label_distribution(ref_data)
        if not dist_df.empty:
            out_path = args.out_dir / "dataset_label_distribution.csv"
            dist_df.to_csv(out_path, index=False)
            print(f"Dataset label distribution (from {first_model_dir.name}, ground truth is model-independent):")
            print(dist_df.to_string(index=False))
            print(f"\nWrote {out_path}")

    metrics_df = model_metrics(args.results_dir)
    if not metrics_df.empty:
        out_path = args.out_dir / "model_metrics.csv"
        metrics_df.to_csv(out_path, index=False)
        print(f"\nCross-model metrics:")
        print(metrics_df.to_string(index=False))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
