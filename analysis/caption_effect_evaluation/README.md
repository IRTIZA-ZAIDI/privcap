# Image-Ground-Truth Caption Effect Evaluation

This directory contains a controlled re-evaluation of three input conditions against the same image-only ground truth:

1. Image only, using the replication predictions
2. Image with a private caption
3. Image with a safe caption

The purpose is to measure how captions change a model's ability to recognize privacy information already present in the image. A private caption may repeat an image-visible category and improve its recognition, but it may also distract the model or introduce categories that are not supported by the image.

## Files

- `evaluate_image_only_ground_truth.py`: reproducible evaluation script
- `image_only_ground_truth_results.json`: complete scores, supports, per-category metrics, selected temperatures, and score changes

## Evaluation design

- All three conditions use only the source image labels as ground truth.
- Comparisons use the intersection of image IDs available in all three conditions and both temperatures.
- The evaluated subsets contain 7,995 VISPR images, 1,550 PrivacyAlert images, and 265 DIPA2 images.
- For each model, dataset, and task, temperature is selected using image-only Macro F1. The selected temperature is then held fixed for both caption conditions.
- T1 and T2 use Macro F1 across the Private and Safe states.
- T3 uses the existing category-wise Macro F1 calculation. The Safe state is included when the dataset contains images with no mapped privacy category.
- DIPA2 has comparable predictions only for T3.

Each result below is reported as `image only / private caption / safe caption`. Values are Macro F1 percentages. The selected temperature appears in parentheses.

## VISPR results

Evaluation support: 7,995 images.

| Model | T1 | T2 | T3 |
|---|---:|---:|---:|
| Ministral-3B | 27.74 / 27.45 / 27.39 (0.1) | 64.41 / 53.97 / 56.34 (0.1) | 34.78 / 37.61 / 39.88 (0.1) |
| Gemma-3-4B | 63.65 / 51.02 / 54.84 (0.1) | 64.27 / 58.84 / 54.92 (1.0) | 39.05 / 31.01 / 33.39 (0.1) |
| Qwen3-VL-8B | 32.82 / 33.83 / 33.87 (0.1) | 41.33 / 43.28 / 43.32 (0.1) | 47.91 / 33.25 / 37.06 (0.1) |
| LLaMA-3.2-11B | 55.81 / 55.34 / 51.57 (1.0) | 44.41 / 39.37 / 36.84 (1.0) | 31.08 / 25.75 / 27.31 (0.1) |

Mean changes relative to image-only evaluation:

| Task | Private caption | Safe caption | Private minus safe |
|---|---:|---:|---:|
| T1 | -3.10 | -3.09 | -0.01 |
| T2 | -4.74 | -5.75 | +1.01 |
| T3 | -6.30 | -3.79 | -2.50 |

## PrivacyAlert results

Evaluation support: 1,550 images.

| Model | T1 | T2 | T3 |
|---|---:|---:|---:|
| Ministral-3B | 44.69 / 44.12 / 43.59 (0.1) | 72.96 / 60.29 / 63.66 (1.0) | 14.79 / 11.49 / 12.66 (0.1) |
| Gemma-3-4B | 68.05 / 68.83 / 69.78 (0.1) | 62.07 / 70.14 / 79.96 (1.0) | 9.09 / 8.33 / 7.58 (1.0) |
| Qwen3-VL-8B | 52.99 / 58.67 / 57.94 (0.1) | 65.06 / 63.40 / 61.70 (0.1) | 13.83 / 11.24 / 12.15 (0.1) |
| LLaMA-3.2-11B | 44.22 / 61.22 / 60.78 (0.1) | 69.19 / 66.87 / 63.04 (1.0) | 9.73 / 10.73 / 9.53 (1.0) |

Mean changes relative to image-only evaluation:

| Task | Private caption | Safe caption | Private minus safe |
|---|---:|---:|---:|
| T1 | +5.72 | +5.54 | +0.19 |
| T2 | -2.15 | -0.23 | -1.92 |
| T3 | -1.41 | -1.38 | -0.03 |

PrivacyAlert T3 is recalculated using one consistent 13-category image-ground-truth mapping across all models. Its values therefore differ from older replication summaries that used inconsistent model-specific T3 target mappings.

## DIPA2 results

Evaluation support: 265 images. Only T3 is available in both the replication and caption-conditioned exports.

| Model | T3 |
|---|---:|
| Ministral-3B | 26.38 / 32.64 / 29.57 (0.1) |
| Gemma-3-4B | 34.54 / 28.76 / 28.37 (0.1) |
| Qwen3-VL-8B | 40.58 / 27.54 / 25.29 (0.1) |
| LLaMA-3.2-11B | 24.30 / 29.70 / 27.02 (0.1) |

Mean T3 changes relative to image-only evaluation:

| Private caption | Safe caption | Private minus safe |
|---:|---:|---:|
| -1.79 | -3.89 | +2.10 |

## Main observations

- Caption effects are strongly model-dependent. Adding text does not consistently improve image-grounded privacy recognition.
- Private captions can help when they repeat information visible in the image. Across VISPR models, Biometric Data F1 increases by 28.27 points on average. Across DIPA2 models, private captions increase Biometric Data by 9.92 points, Vehicle Information by 5.06 points, and Location Identifiers by 4.45 points.
- These category gains can be offset by source interference. Private captions often reduce Safe-state F1 or prompt categories that are not supported by the image. VISPR T3 consequently falls by 6.30 points on average despite the biometric improvement.
- DIPA2 provides the clearest aggregate indication of useful repeated information: private captions outperform safe captions by 2.10 points on average. This effect remains model-dependent because only Ministral and LLaMA improve over their image-only scores.
- PrivacyAlert T1 improves by similar amounts with private and safe captions. This suggests a general caption-conditioning effect rather than a benefit caused specifically by privacy-disclosing text.

## Run the evaluation

The script resolves its inputs relative to `ROOT = SCRIPT_DIR.parents[2]` and expects
a `Full Runs/` directory at that location containing the raw per-model prediction
JSON files (both the image-only replication runs and the `_Multimodal` caption-conditioned
runs). Those raw files are not committed to this repository — they are the same
multi-gigabyte outputs referenced in the main [README](../../README.md), hosted on
the Hugging Face dataset repo rather than in git.

To reproduce:

1. Obtain the `Full Runs/` tree (VISPR/PrivacyAlert/DIPA2 replication results plus
   `_Multimodal/<dataset>/<model>/` caption-conditioned results) from the Hugging
   Face dataset repo once published, or from local working data if you generated it
   from the notebooks under `notebooks/evaluation/`.
2. Place or symlink that tree at the repository root as `Full Runs/`, matching the
   layout `evaluate_image_only_ground_truth.py` expects (see `MULTIMODAL_FILES` and
   `REPLICATION_FILES` at the top of the script for the exact paths).
3. From this directory, run:

```bash
python evaluate_image_only_ground_truth.py
```

The script prints the summary tables and rewrites `image_only_ground_truth_results.json`
with the complete auditable results.

`image_only_ground_truth_results.json` in this directory is the already-computed,
committed output — you do not need the raw `Full Runs/` tree just to read the results
above or in the JSON.
