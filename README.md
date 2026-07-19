# privcap — dataset drive index

Central index of where this project's datasets live. The heavy data (raw
images, caption sets, label files) stays in **Google Drive**; this repo only
records **where** each piece is, so anyone can find and fetch it.

> ⚠️ The data itself is intentionally **not** committed here — only links +
> metadata. Keep large files out of git.

## Datasets

| Dataset | Page |
|---|---|
| VISPR | [datasets/vispr](datasets/vispr/README.md) |
| PrivacyAlert | [datasets/privacy_alert](datasets/privacy_alert/README.md) |
| DIPA2 | [datasets/dipa2](datasets/dipa2/README.md) |

## What each dataset page records

| Content | Meaning |
|---|---|
| **Raw images & labels** | Original dataset as downloaded |
| **Captions (both)** | Generated caption sets (both variants) |
| **Cleaned labels** | Post-audit / cleaned label files |
| **Caption-gen notebook** | Notebook used to produce the captions |

## Adding / updating a link

1. Open the dataset's `README.md` under `datasets/<name>/`.
2. Paste the Drive **share URL** into the `Link` column.
3. Leave the folder name in the `Drive folder` column so it's easy to locate
   in Drive.
4. `links.yaml` is the machine-readable mirror (for notebooks/scripts) — keep
   it in sync with the tables.
