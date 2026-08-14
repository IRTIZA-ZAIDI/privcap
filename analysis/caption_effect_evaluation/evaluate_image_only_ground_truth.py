#!/usr/bin/env python3
"""Re-evaluate image-only and image-caption predictions against image ground truth.

The script compares three input conditions on the same image IDs:
  1. image only (replication),
  2. image + private caption, and
  3. image + safe caption.

For each model, dataset, and task, the temperature is selected once from the
image-only condition by Macro F1 and then held fixed for both caption conditions.
This isolates the effect of adding a caption from temperature selection and from
changes to the ground truth.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]

MODELS = {
    "ministral": "Ministral-3B",
    "gemma": "Gemma-3-4B",
    "qwen": "Qwen3-VL-8B",
    "llama": "LLaMA-3.2-11B",
}

MULTIMODAL_FILES = {
    "VISPR": {
        "ministral": ROOT / "Full Runs/_Multimodal/VISPR/ministral/mistralai_ministral-3-3b_vispr_multimodal_20260624_170215_CORRECTED.json",
        "gemma": ROOT / "Full Runs/_Multimodal/VISPR/gemma/google_gemma-3-4b_vispr_multimodal_20260629_094855_CORRECTED.json",
        "qwen": ROOT / "Full Runs/_Multimodal/VISPR/qwen/qwen_qwen3-vl-8b_vispr_multimodal_20260622_155902_CORRECTED.json",
        "llama": ROOT / "Full Runs/_Multimodal/VISPR/llama/meta_llama3_2-vl-11b_vispr_multimodal_20260801_070059_CORRECTED.json",
    },
    "PrivacyAlert": {
        "ministral": ROOT / "Full Runs/_Multimodal/PrivacyAlert/ministral/mistralai_ministral-3-3b_privacyalert_multimodal_20260807_064809_all_tasks.json",
        "gemma": ROOT / "Full Runs/_Multimodal/PrivacyAlert/gemma/google_gemma-3-4b_privacyalert_multimodal_20260809_091539_all_tasks.json",
        "qwen": ROOT / "Full Runs/_Multimodal/PrivacyAlert/qwen/qwen_qwen3-vl-8b_privacyalert_multimodal_20260801_215526_all_tasks.json",
        "llama": ROOT / "Full Runs/_Multimodal/PrivacyAlert/llama/meta_llama3_2-vl-11b_privacyalert_multimodal_20260810_082052_all_tasks.json",
    },
    "DIPA2": {
        "ministral": ROOT / "Full Runs/_Multimodal/DIPA2/ministral/mistralai_ministral-3-3b_dipa2_multimodal_20260813_021101_all_tasks.json",
        "gemma": ROOT / "Full Runs/_Multimodal/DIPA2/gemma/google_gemma-3-4b_dipa2_multimodal_20260813_000237_all_tasks.json",
        "qwen": ROOT / "Full Runs/_Multimodal/DIPA2/qwen/qwen_qwen3-vl-8b_dipa2_multimodal_20260812_203503_all_tasks.json",
        "llama": ROOT / "Full Runs/_Multimodal/DIPA2/llama/meta_llama3-2-11b_dipa2_multimodal_20260813_045224_all_tasks.json",
    },
}

REPLICATION_FILES = {
    "VISPR": {
        "ministral": ROOT / "Full Runs/VISPR/ministral/mistralai_Ministral-3-3B-Instruct-2512_20260414_184228_results.json",
        "qwen": ROOT / "Full Runs/VISPR/qwen/Qwen_Qwen3-VL-8B-Instruct_20260416_151303_results.json",
        "llama": ROOT / "Full Runs/VISPR/llama/meta-llama_Llama-3_2-11B-Vision-Instruct_20260523_113548_results.json",
    },
    "PrivacyAlert": {
        "ministral": ROOT / "Full Runs/PrivacyAlert/ministral/mistralai_Ministral-3-3B-Instruct-2512-BF16_privacyalert_20260523_100634_results.json",
        "gemma": ROOT / "Full Runs/PrivacyAlert/gemma/google_gemma-3-4b-it_privacyalert_20260523_042225_results.json",
        "qwen": ROOT / "Full Runs/PrivacyAlert/qwen/Qwen_Qwen3-VL-8B-Instruct_privacyalert_20260522_194006_results.json",
        "llama": ROOT / "Full Runs/PrivacyAlert/llama/meta-llama_Llama-3_2-11B-Vision-Instruct_privacyalert_20260521_024216_results.json",
    },
    "DIPA2": {
        "ministral": ROOT / "Full Runs/DIPA2/ministral/mistralai_Ministral-3-3B-Instruct-2512-BF16_dipa2_20260521_181043_results.json",
        "gemma": ROOT / "Full Runs/DIPA2/gemma/google_gemma-3-4b-it_dipa2_20260520_144522_results.json",
        "qwen": ROOT / "Full Runs/DIPA2/qwen/Qwen_Qwen3-VL-8B-Instruct_dipa2_20260520_104048_results.json",
        "llama": ROOT / "Full Runs/DIPA2/llama/meta-llama_Llama-3_2-11B-Vision-Instruct_dipa2_20260520_155350_results.json",
    },
}

VISPR_GEMMA_REPLICATION = {
    0.1: {
        "task1": [ROOT / "Full Runs/VISPR/gemma/gemma-4b-0.1-20260428T111055Z-3-001/gemma-4b-0.1/google_gemma-3-4b-it_20260427_183948_ckpt_0_1_task1.json"],
        "task2": [ROOT / "Full Runs/VISPR/gemma/gemma-4b-0.1-20260428T111055Z-3-001/gemma-4b-0.1/google_gemma-3-4b-it_20260427_183948_ckpt_0_1_task2.json"],
        "task3": [ROOT / "Full Runs/VISPR/gemma/gemma-4b-0.1-20260428T111055Z-3-001/gemma-4b-0.1/google_gemma-3-4b-it_20260427_185655_ckpt_0_1_task3.json"],
    },
    1.0: {
        "task1": [ROOT / "Full Runs/VISPR/gemma/gemma-4b-20260425T095936Z-3-001/gemma-4b/google_gemma-3-4b-it_20260424_083728_ckpt_1_0_task1.json"],
        "task2": [ROOT / "Full Runs/VISPR/gemma/gemma-4b-20260425T095936Z-3-001/gemma-4b/google_gemma-3-4b-it_20260424_083728_ckpt_1_0_task2.json"],
        "task3": [
            ROOT / "Full Runs/VISPR/gemma/gemma-4b-20260425T095936Z-3-001/gemma-4b/google_gemma-3-4b-it_20260424_083728_ckpt_1_0_task3.json",
            ROOT / "Full Runs/VISPR/gemma/gemma-4b-20260425T095936Z-3-001/gemma-4b/google_gemma-3-4b-it_20260425_085251_ckpt_1_0_task3.json",
        ],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def temperature(key: str | float) -> float:
    return float(str(key).replace("temp=", ""))


def index_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = str(record["id"])
        if item_id in indexed and indexed[item_id] != record:
            raise ValueError(f"Conflicting duplicate record for {item_id}")
        indexed[item_id] = record
    return indexed


def load_replication(dataset: str, model: str) -> dict[str, dict[float, dict[str, dict[str, Any]]]]:
    """Return task -> temperature -> id -> result record."""
    output: dict[str, dict[float, dict[str, dict[str, Any]]]] = defaultdict(dict)
    if dataset == "VISPR" and model == "gemma":
        for temp, tasks in VISPR_GEMMA_REPLICATION.items():
            for task, paths in tasks.items():
                merged: list[dict[str, Any]] = []
                for path in paths:
                    merged.extend(load_json(path)["results"])
                output[task][temp] = index_records(merged)
        return dict(output)

    raw = load_json(REPLICATION_FILES[dataset][model])
    for temp_key, task_data in raw.items():
        if not str(temp_key).startswith("temp="):
            continue
        temp = temperature(temp_key)
        for task, block in task_data.items():
            if isinstance(block, dict) and "results" in block:
                output[task][temp] = index_records(block["results"])
    return dict(output)


def load_multimodal(path: Path) -> dict[str, dict[float, dict[str, dict[str, dict[str, Any]]]]]:
    """Return task -> temperature -> condition -> id -> result record."""
    raw = load_json(path)
    output: dict[str, dict[float, dict[str, dict[str, dict[str, Any]]]]] = defaultdict(dict)
    for task, task_data in raw.items():
        if not task.startswith("task") or not isinstance(task_data, dict):
            continue
        for temp_key, temp_data in task_data.items():
            temp = temperature(temp_key)
            conditions: dict[str, dict[str, dict[str, Any]]] = {}
            for condition in ("explicit", "no_leak"):
                if condition in temp_data and "results" in temp_data[condition]:
                    conditions[condition] = index_records(temp_data[condition]["results"])
            output[task][temp] = conditions
    return dict(output)


def image_target(record: dict[str, Any], task: str) -> str | frozenset[str]:
    if task in ("task1", "task2"):
        value = str(record["gt"]).strip().lower()
        return "Private" if value == "private" else "Safe"
    labels = record.get("gt_labels", record.get("gt_combined", []))
    return frozenset(str(label) for label in labels if str(label).lower() != "safe")


def prediction(record: dict[str, Any], task: str) -> str | frozenset[str]:
    if task in ("task1", "task2"):
        value = str(record["prediction"]).strip().lower()
        return "Private" if value == "private" else "Safe"
    return frozenset(str(label) for label in record.get("pred_labels", []) if str(label).lower() != "safe")


def f1_for_class(y_true: list[Any], y_pred: list[Any], label: Any) -> tuple[float, float, float, int]:
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
    support = sum(t == label for t in y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, support


def score_binary(
    targets: dict[str, str | frozenset[str]],
    records: dict[str, dict[str, Any]],
    ids: list[str],
    task: str,
) -> dict[str, Any]:
    y_true = [targets[item_id] for item_id in ids]
    y_pred = [prediction(records[item_id], task) for item_id in ids]
    per_class = {}
    for label in ("Private", "Safe"):
        p, r, f1, support = f1_for_class(y_true, y_pred, label)
        per_class[label] = {
            "precision": round(100 * p, 2),
            "recall": round(100 * r, 2),
            "f1": round(100 * f1, 2),
            "support": support,
        }
    accuracy = mean(t == p for t, p in zip(y_true, y_pred))
    return {
        "macro_precision": round(mean(v["precision"] for v in per_class.values()), 2),
        "macro_recall": round(mean(v["recall"] for v in per_class.values()), 2),
        "macro_f1": round(mean(v["f1"] for v in per_class.values()), 2),
        "accuracy": round(100 * accuracy, 2),
        "n": len(ids),
        "class_metrics": per_class,
    }


def score_multilabel(
    targets: dict[str, str | frozenset[str]],
    records: dict[str, dict[str, Any]],
    ids: list[str],
    task: str,
) -> dict[str, Any]:
    gt_sets = [targets[item_id] for item_id in ids]
    pred_sets = [prediction(records[item_id], task) for item_id in ids]
    categories = sorted({label for labels in gt_sets for label in labels})
    per_category = {}

    for category in categories:
        y_true = [category in labels for labels in gt_sets]
        y_pred = [category in labels for labels in pred_sets]
        p, r, f1, support = f1_for_class(y_true, y_pred, True)
        per_category[category] = {
            "precision": round(100 * p, 2),
            "recall": round(100 * r, 2),
            "f1": round(100 * f1, 2),
            "support": support,
        }

    safe_support = sum(not labels for labels in gt_sets)
    if safe_support:
        y_true = [not labels for labels in gt_sets]
        y_pred = [not labels for labels in pred_sets]
        p, r, f1, support = f1_for_class(y_true, y_pred, True)
        per_category["Safe"] = {
            "precision": round(100 * p, 2),
            "recall": round(100 * r, 2),
            "f1": round(100 * f1, 2),
            "support": support,
        }

    return {
        "macro_precision": round(mean(v["precision"] for v in per_category.values()), 2),
        "macro_recall": round(mean(v["recall"] for v in per_category.values()), 2),
        "macro_f1": round(mean(v["f1"] for v in per_category.values()), 2),
        "n": len(ids),
        "image_category_assignments": sum(len(labels) for labels in gt_sets),
        "category_metrics": per_category,
    }


def score(
    task: str,
    targets: dict[str, str | frozenset[str]],
    records: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    if task in ("task1", "task2"):
        return score_binary(targets, records, ids, task)
    return score_multilabel(targets, records, ids, task)


def reference_targets(
    dataset: str,
    task: str,
    multimodal_by_model: dict[str, dict[str, dict[float, dict[str, dict[str, dict[str, Any]]]]]],
) -> dict[str, str | frozenset[str]]:
    references: list[tuple[str, dict[str, str | frozenset[str]]]] = []
    for model, multimodal in multimodal_by_model.items():
        if task not in multimodal:
            continue
        first_temp = sorted(multimodal[task])[0]
        records = multimodal[task][first_temp]["no_leak"]
        references.append((model, {item_id: image_target(record, task) for item_id, record in records.items()}))
    if not references:
        raise ValueError(f"No image ground truth found for {dataset} {task}")

    source_model, source = references[0]
    for model, candidate in references[1:]:
        common = set(source) & set(candidate)
        mismatches = [item_id for item_id in common if source[item_id] != candidate[item_id]]
        if mismatches:
            raise ValueError(
                f"Image-ground-truth mismatch for {dataset} {task}: "
                f"{source_model} vs {model}, first ID {mismatches[0]}"
            )
    return source


def evaluate_all() -> dict[str, Any]:
    final: dict[str, Any] = {"method": {}, "results": []}
    final["method"] = {
        "ground_truth": "Image labels only for all three input conditions",
        "pairing": "Intersection of image IDs available in every compared condition and temperature",
        "temperature_selection": "Highest image-only Macro F1; same temperature applied to both caption conditions",
        "conditions": {
            "image_only": "Replication prediction",
            "private_caption": "Multimodal explicit/private-caption prediction",
            "safe_caption": "Multimodal no-leak/safe-caption prediction",
        },
    }

    for dataset, model_paths in MULTIMODAL_FILES.items():
        multimodal_by_model = {model: load_multimodal(path) for model, path in model_paths.items()}
        tasks = ("task1", "task2", "task3") if dataset != "DIPA2" else ("task3",)
        targets_by_task = {
            task: reference_targets(dataset, task, multimodal_by_model) for task in tasks
        }

        for model in MODELS:
            replication = load_replication(dataset, model)
            multimodal = multimodal_by_model[model]
            for task in tasks:
                common_temps = sorted(set(replication[task]) & set(multimodal[task]))
                if not common_temps:
                    raise ValueError(f"No shared temperature for {dataset} {model} {task}")

                id_sets: list[set[str]] = [set(targets_by_task[task])]
                for temp in common_temps:
                    id_sets.extend(
                        [
                            set(replication[task][temp]),
                            set(multimodal[task][temp]["explicit"]),
                            set(multimodal[task][temp]["no_leak"]),
                        ]
                    )
                common_ids = sorted(set.intersection(*id_sets))
                if not common_ids:
                    raise ValueError(f"No common IDs for {dataset} {model} {task}")

                image_only_by_temp = {
                    temp: score(task, targets_by_task[task], replication[task][temp], common_ids)
                    for temp in common_temps
                }
                selected_temp = min(
                    common_temps,
                    key=lambda temp: (-image_only_by_temp[temp]["macro_f1"], temp),
                )

                condition_scores = {
                    "image_only": image_only_by_temp[selected_temp],
                    "private_caption": score(
                        task,
                        targets_by_task[task],
                        multimodal[task][selected_temp]["explicit"],
                        common_ids,
                    ),
                    "safe_caption": score(
                        task,
                        targets_by_task[task],
                        multimodal[task][selected_temp]["no_leak"],
                        common_ids,
                    ),
                }
                base_f1 = condition_scores["image_only"]["macro_f1"]
                final["results"].append(
                    {
                        "dataset": dataset,
                        "model": MODELS[model],
                        "model_key": model,
                        "task": task.upper().replace("TASK", "T"),
                        "temperature": selected_temp,
                        "n": len(common_ids),
                        "scores": condition_scores,
                        "delta_private_vs_image": round(
                            condition_scores["private_caption"]["macro_f1"] - base_f1, 2
                        ),
                        "delta_safe_vs_image": round(
                            condition_scores["safe_caption"]["macro_f1"] - base_f1, 2
                        ),
                        "delta_private_vs_safe": round(
                            condition_scores["private_caption"]["macro_f1"]
                            - condition_scores["safe_caption"]["macro_f1"],
                            2,
                        ),
                    }
                )
    return final


def print_summary(report: dict[str, Any]) -> None:
    rows = report["results"]
    for dataset in MULTIMODAL_FILES:
        print(f"\n{dataset} (Macro F1 %, image ground truth for every condition)")
        print("Model | Task | Temp | n | Image only | + Private caption | + Safe caption | Delta private | Delta safe")
        print("---|---:|---:|---:|---:|---:|---:|---:|---:")
        for row in rows:
            if row["dataset"] != dataset:
                continue
            scores = row["scores"]
            print(
                f'{row["model"]} | {row["task"]} | {row["temperature"]:g} | {row["n"]} | '
                f'{scores["image_only"]["macro_f1"]:.2f} | '
                f'{scores["private_caption"]["macro_f1"]:.2f} | '
                f'{scores["safe_caption"]["macro_f1"]:.2f} | '
                f'{row["delta_private_vs_image"]:+.2f} | {row["delta_safe_vs_image"]:+.2f}'
            )

    print("\nMean change across models (percentage points)")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["task"])].append(row)
    for (dataset, task), items in grouped.items():
        print(
            f"{dataset} {task}: private {mean(x['delta_private_vs_image'] for x in items):+.2f}, "
            f"safe {mean(x['delta_safe_vs_image'] for x in items):+.2f}, "
            f"private-minus-safe {mean(x['delta_private_vs_safe'] for x in items):+.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "image_only_ground_truth_results.json",
        help="Path for the complete JSON report",
    )
    args = parser.parse_args()
    report = evaluate_all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print_summary(report)
    print(f"\nDetailed report: {args.output}")


if __name__ == "__main__":
    main()
