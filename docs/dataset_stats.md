# Dataset statistics

Summary numbers from the current benchmark. Full per-category breakdowns,
per-model result tables, and the complete dataset card will be published on
the Hugging Face dataset repo (see the main [README](../README.md)). This
page holds the lightweight, browsable summary.

## Benchmark at a glance

| Source dataset | Images | Private images | Safe/Public images | Private captions | Safe captions |
|---|---:|---:|---:|---:|---:|
| VISPR | 7,995 | 4,993 | 3,002 | 7,995 | 7,995 |
| PrivacyAlert | 1,550 | 366 | 1,184 | 1,550 | 1,550 |
| DIPA2 | 853 | 853 | 0 | 853 | 853 |
| **Total** | **10,398** | **6,212** | **4,186** | **10,398** | **10,398** |

The complete caption corpus contains 20,796 captions. DIPA2 generation covers
853 images; the reported evaluation uses the 265-image unanimous-agreement
subset.

## Privacy-category composition by modality

Cell values are `percentage (raw count)`. Source-image labels vs. private-caption
labels, per dataset. Safe captions carry no target privacy label and are excluded.

| Privacy category | VISPR image | VISPR private caption | PrivacyAlert image | PrivacyAlert private caption | DIPA2 image | DIPA2 private caption |
|---|---:|---:|---:|---:|---:|---:|
| Financial Information | 0.3% (121) | 1.4% (269) | 0.0% (0) | 0.1% (1) | - | - |
| Nudity | 1.2% (460) | 0.9% (166) | 28.6% (654) | 6.1% (51) | - | - |
| Vehicle Information | 0.7% (271) | 2.5% (480) | 2.1% (49) | 2.3% (19) | 8.8% (96) | 4.5% (97) |
| HIPAA Data | 0.6% (225) | 2.6% (507) | 3.4% (78) | 2.0% (17) | - | - |
| Location Identifiers | 3.3% (1,298) | 12.3% (2,366) | 0.0% (1) | 0.0% (0) | 9.3% (102) | 13.0% (277) |
| Personal Context | 6.2% (2,428) | 31.3% (6,002) | 15.2% (348) | 32.6% (273) | 34.3% (375) | 37.7% (804) |
| Digital Identifiers | 0.5% (209) | 1.2% (239) | 5.3% (121) | 3.2% (27) | 10.8% (118) | 15.9% (339) |
| Personal Metadata (Demographics) | 33.6% (13,081) | 25.8% (4,953) | 11.0% (250) | 19.2% (161) | - | - |
| Background Individuals | 1.5% (565) | 1.5% (282) | 7.5% (171) | 3.6% (30) | 0.9% (10) | 0.8% (17) |
| Legal Identifiers | 5.6% (2,181) | 13.9% (2,671) | 3.1% (70) | 11.0% (92) | 4.8% (52) | 4.7% (100) |
| Biometric Data | 46.3% (18,015) | 4.2% (807) | 8.8% (201) | 5.7% (48) | 29.6% (324) | 22.2% (475) |
| Violent/Unlawful Actions | 0.1% (20) | 2.4% (458) | 11.1% (253) | 8.5% (71) | 1.6% (17) | 1.2% (26) |
| Children | - | - | 3.8% (87) | 5.7% (48) | - | - |

## Reproducing these numbers / computing more

`scripts/dataset_stats/compute_result_stats.py` reads any dataset's
`*_all_tasks.json` evaluation output and derives:

- `dataset_label_distribution.csv`: category support counts per task/condition
  (ground truth, so this is a dataset property, not a model score).
- `model_metrics.csv`: macro F1/precision/recall per model, task, temperature,
  and condition, for cross-model comparison.

```bash
pip install -r scripts/dataset_stats/requirements.txt
python scripts/dataset_stats/compute_result_stats.py \
  --results-dir /path/to/VISPR \
  --out-dir ./out
```

`--results-dir` must contain one subdirectory per model, each with a
`*_all_tasks.json` file (the format written by the notebooks under
`notebooks/evaluation/<dataset>/<model>/`).

## Status

This page currently mirrors the summary tables reported in the paper draft.
Additional stats and the full dataset card are pending and will be attached
here and/or on the Hugging Face dataset repo.
