# Kaggle — Squares in Squares

Code for <https://www.kaggle.com/competitions/squares-in-squares>: pack n unit squares
(n = 1…1000) into the smallest axis-aligned square. Score per puzzle is `L/√n − 1`, summed
(lower is better).

## Pipeline

| Script | What it does |
|---|---|
| `build_submission.py` | v1 baseline: best of trivial grid and a Göbel-style 45° diagonal-strip packing per n. |
| `parse_records.py` | Extracts exact coordinates from David Ellsworth's record SVGs (`refs/svg`) into `cands/records.json`. |
| `merge.py` | Merges a base CSV with candidate JSONs, keeps the smallest valid side per n, and applies the drop-a-square sweep (s(n) ≤ s(n+1)). |
| `check_submission.py` | Local feasibility check (strict SAT gaps) + expected score. |

Exact contacts are turned into certified gaps by scaling every packing by `1 + 1e-12`, since the
metric certifies non-overlap with interval arithmetic.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python build_submission.py submission.csv
mkdir -p refs/svg  # download record SVGs from https://kingbird.myphotos.cc/packing/squares_in_squares.html
.venv/bin/python parse_records.py refs/svg cands/records.json
.venv/bin/python merge.py submission.csv submission_v2.csv cands/records.json
.venv/bin/kaggle competitions submit -c squares-in-squares -f submission_v2.csv -m "..."
```

## Scores

| Version | Public LB |
|---|---|
| v1 grid + diagonal strip | 24.02152 |
