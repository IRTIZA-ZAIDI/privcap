# Data sources

PRIVCAP is built from three existing visual-privacy datasets. This repo does
not redistribute their raw images. See [Responsible use](../README.md#responsible-use-and-data-governance)
in the main README. Obtain each source dataset from its own official release
and license terms:

| Source dataset | Accessible images used | Official source |
|---|---:|---|
| VISPR | 7,995 | Orekondy et al., 2017; request access via the [VISPR project page](https://github.com/orekondy/vispr) |
| PrivacyAlert | 1,550 | Zhao et al., 2022 |
| DIPA2 | 853 | Xu et al., 2023 |

Some original source URLs were unavailable at the time of construction; the
accessible subsets used here are identified by the caption records and
evaluation files in this repo and on the Hugging Face dataset.

## Working Drive folders (team reference)

Raw images, intermediate label-cleaning runs, and per-model result dumps
were staged in shared Google Drive folders during development. These are
internal working folders, not a stable public distribution. The durable,
citable copy is (or will be) the Hugging Face dataset repo linked from the
main README.

| Dataset | Working folder |
|---|---|
| VISPR | https://drive.google.com/drive/folders/1gqF7zRYxnec15nHJNnHXYIkojNg8vya1 |
| PrivacyAlert | https://drive.google.com/drive/folders/17af2kx5TZMjPPrxGttYyX0cKh2R6iTyV |
| DIPA2 | https://drive.google.com/drive/folders/1ujhNttc9_GoODebCyyDjDr5-b50PXh3X |
