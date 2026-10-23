# Kaggle — Squares in Squares

Code for <https://www.kaggle.com/competitions/squares-in-squares>: pack n unit squares
(n = 1…1000) into the smallest axis-aligned square. Score per puzzle is `L/√n − 1`, summed
(lower is better).

## Pipeline

| Script | What it does |
|---|---|
| `build_submission.py` | v1 baseline: best of trivial grid and a Göbel-style 45° diagonal-strip packing per n. |
| `parse_records.py` | Extracts exact coordinates from David Ellsworth's record SVGs (`refs/svg`) into `cands/records.json`. |
| `parse_certs.py` | Converts Francisco Couzo's exact rational certificates (`franciscouzo/square-packing`, `x y tan(θ/2)`) into `cands/couzo.json`. |
| `parse_witnesses.py` | Converts `jlevy/squares` witness YAMLs (corners / center-basis / center-angle) into `cands/jlevy.json`. |
| `families.py` | Closed-form Göbel-square packings (45° b×b block + four staircases). |
| `merge.py` | Merges a base CSV with candidate JSONs, keeps the smallest valid side per n, then alternates L-extension (grow side by m, fill the L strip with grid squares) and drop-a-square (s(n) ≤ s(n+1)) passes. Writes `<out>.sources.json`. |
| `optimizer.py` | Local compaction: shrink the box, minimize squared SAT penetration + wall violation with L-BFGS, basin-hopping kicks, strict legalization. |
| `metric/metric.py` | Official competition metric v17 (sha256 `312458…9733`), for scoring locally exactly as Kaggle does. |
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
| v2 + Ellsworth records, Göbel squares, Couzo certificates, L-extension/drop | 22.64887 |

## References (local, git-ignored under `refs/`)

- `refs/ellsworth_squares_in_squares.html`, `refs/svg/` — David Ellsworth's record page + all SVGs
- `refs/square-packing/` — franciscouzo/square-packing (certificates through n=10000)
- `refs/jlevy_squares/` (sparse: `packing/witnesses`, `packing/frontier`), `refs/evand_square_packing/`
- `refs/papers/` — Arslanov et al. 2021 (n(n−1) squares), Friedman survey
- `refs/notebooks/` — public Kaggle notebooks (fault-line DP, per-puzzle leaderboard)
