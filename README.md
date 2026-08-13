# PRIVCAP

PRIVCAP is a benchmark for studying privacy detection in vision-language
models (VLMs). It pairs each real-world image with two matched, first-person
social-media-style captions:

- a **private caption** that explicitly discloses at least one privacy attribute;
- a **safe caption** that describes the same broad context without introducing
  a target disclosure.

Holding the image and posting context fixed makes it possible to measure how
privacy-bearing text changes model behavior, and whether a model attributes a
disclosure to the image, the caption, or both.

The benchmark is built from accessible images in [VISPR](https://github.com/orekondy/vispr),
PrivacyAlert, and DIPA2. Fine-grained source and caption labels are mapped to
a shared 14-category privacy taxonomy for evaluation. See
[docs/dataset_stats.md](docs/dataset_stats.md) for the full breakdown.

## What's in this repo

This repo holds the **code**: caption-generation notebooks, evaluation
notebooks, and analysis scripts. It does not hold the raw images or the
multi-gigabyte evaluation result files. Those are hosted on the
**Hugging Face dataset repo** (link pending publication) so they can be
browsed, versioned, and loaded directly with `datasets.load_dataset()`
instead of bloating this git history. See [docs/data_sources.md](docs/data_sources.md)
for where the source images come from and where working data currently lives.

```text
.
├── notebooks/
│   ├── caption_generation/     # generates matched private/safe caption pairs per dataset
│   │   ├── vispr_caption_generation.ipynb
│   │   ├── privacyalert_caption_generation.ipynb
│   │   └── dipa2_caption_generation.ipynb
│   └── evaluation/             # per-dataset, per-model VLM evaluation notebooks (T1-T4)
│       ├── vispr/{gemma,llama,ministral,qwen}/
│       ├── privacyalert/{gemma,llama,ministral,qwen}/
│       └── dipa2/dipa2_multimodal_all_models.ipynb
├── scripts/
│   └── dataset_stats/          # compute label distributions + cross-model metrics from result JSON
├── docs/
│   ├── dataset_stats.md        # benchmark summary tables
│   └── data_sources.md         # where the source datasets and working data live
└── README.md
```

## Models evaluated

| Paper name | Hugging Face checkpoint |
|---|---|
| Ministral-3B | `mistralai/Ministral-3-3B-Instruct-2512-BF16` |
| Gemma-3-4B | `google/gemma-3-4b-it` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` |
| LLaMA-3.2-11B | `meta-llama/Llama-3.2-11B-Vision-Instruct` |

Runs use three fixed seeds at temperatures 0.1 and 1.0. Binary outputs are
aggregated by majority vote. See the evaluation notebooks for the complete
prompt, parsing, model-loading, and metric definitions.

## Evaluation tasks

| Task | Name | Output |
|---|---|---|
| T1 | Binary Privacy Detection | One Private/Safe judgment |
| T2 | Taxonomy-Guided State Classification | One Private/Safe judgment with the taxonomy provided in the prompt |
| T3 | Multi-Label Attribute Recognition | All applicable privacy categories |
| T4 | Joint Taxonomy-Guided Source Attribution | Separate Private/Safe judgments for the image and caption |

T1-T3 are evaluated on image-only inputs and on image-caption pairs. T4 is
evaluated on image-caption pairs. DIPA2 evaluation uses T3+T4 only, since all
DIPA2 images are private.

## Reproducing the workflow

### 1. Obtain the source datasets

Request or download VISPR, PrivacyAlert, and DIPA2 from their official
sources and follow their licenses and access conditions. Source images are
not redistributed by this repository. See [docs/data_sources.md](docs/data_sources.md).

### 2. Generate caption pairs

Open the relevant notebook under `notebooks/caption_generation/` in Google
Colab. Each notebook installs its dependencies, loads source annotations and
images, generates the two captions and label anchors, validates the
structured output, and records failures separately. Requires a Google GenAI
API key. Do not commit API keys, tokens, or private dataset credentials.

### 3. Run evaluation

Use the model-specific notebooks under `notebooks/evaluation/<dataset>/<model>/`.
Each notebook defines its model checkpoint, dataset paths, prompts, decoding
settings, parser, metrics, and output directory, and writes JSON (and where
available, XLSX) result files. Update the Drive/image-root paths near the
top of each notebook before running it. The committed notebooks retain the
paths used for the reported experiments.

### 4. Compute stats

```bash
pip install -r scripts/dataset_stats/requirements.txt
python scripts/dataset_stats/compute_result_stats.py \
  --results-dir /path/to/VISPR \
  --out-dir ./out
```

See [docs/dataset_stats.md](docs/dataset_stats.md) for what this produces and
the current summary numbers.

## Hardware and software

The reported evaluation runs were executed in Google Colab using NVIDIA A100
GPUs. Individual notebooks install or import the required packages,
including PyTorch, Transformers, Accelerate, Pillow, NumPy, pandas,
scikit-learn, and openpyxl. Caption generation additionally uses `google-genai`.

Model access can require accepting the corresponding Hugging Face license
and authenticating with a Hugging Face token.

## Responsible use and data governance

This repository is intended for privacy research. The source images remain
governed by the licenses and access policies of VISPR, PrivacyAlert, and
DIPA2. Generated private captions may contain synthetic privacy-bearing text
and should be handled as sensitive research data. Do not use the benchmark
to identify, profile, or target individuals.

## Citation

Citation metadata will be added after the review process.

## Release status

This is a working research repository. Before archival release: the Hugging
Face dataset link, final result tables, a package-version lock file, and a
repository license will be added. The paper is currently under anonymous
review and is intentionally not included in this repository.
