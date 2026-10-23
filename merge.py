"""Merge candidate packings into one submission, keeping the smallest valid side per n.

Sources: a base submission CSV plus any number of candidate JSON files of the form
{n: {"squares": [[x, y, angle], ...]}} (strings or numbers, angle in radians).
Every candidate is translated to the origin, scaled by (1 + EPS) to turn exact
contacts into certified gaps, then checked with check_submission.check.

    python merge.py base.csv out.csv cand1.json [cand2.json ...]
"""

import json
import math
import sys

import mpmath
import pandas as pd

from check_submission import check

mpmath.mp.dps = 50
EPS = mpmath.mpf("1e-12")


def normalize(squares):
    sq = [tuple(mpmath.mpf(str(v)) for v in s) for s in squares]
    xs, ys = [], []
    for x, y, a in sq:
        r = abs(mpmath.cos(a)) / 2 + abs(mpmath.sin(a)) / 2
        xs.append(x - r)
        ys.append(y - r)
    x0, y0 = min(xs), min(ys)
    return [((x - x0) * (1 + EPS), (y - y0) * (1 + EPS), a) for x, y, a in sq]


def fmt(v):
    return mpmath.nstr(v, 32, strip_zeros=False, min_fixed=-10**9, max_fixed=10**9)


def frame(n, sq):
    return pd.DataFrame({"n": n, "square_id": range(n),
                         "x": [fmt(s[0]) for s in sq], "y": [fmt(s[1]) for s in sq],
                         "angle": [fmt(s[2]) for s in sq]})


def main(base, out, cands):
    df = pd.read_csv(base, dtype={"x": str, "y": str, "angle": str})
    best = {}
    for n, g in df.groupby("n"):
        g = g.sort_values("square_id")
        L, msg = check(n, g)
        if L is not None:
            best[n] = (L, g[["n", "square_id", "x", "y", "angle"]], "base")
    for path in cands:
        data = json.load(open(path))
        won = 0
        for key, rec in data.items():
            n = int(key)
            if n > 1000 or len(rec["squares"]) != n:
                continue
            g = frame(n, normalize(rec["squares"]))
            L, msg = check(n, g)
            if L is None:
                print(f"{path} n={n}: rejected ({msg})")
                continue
            if n not in best or L < best[n][0] - 1e-12:
                best[n] = (L, g, path)
                won += 1
        print(f"{path}: improved {won} puzzles")
    # s(n) <= s(n+1): dropping a square from a valid packing keeps it valid.
    dropped = 0
    for n in range(999, 0, -1):
        if n + 1 in best and (n not in best or best[n + 1][0] < best[n][0] - 1e-12):
            g = best[n + 1][1].sort_values("square_id").iloc[:n].copy()
            g["n"] = n
            L, msg = check(n, g)
            if L is not None and (n not in best or L < best[n][0] - 1e-12):
                best[n] = (L, g, f"drop<-{n + 1}")
                dropped += 1
    print(f"drop-a-square sweep: improved {dropped} puzzles")
    parts = [best[n][1] for n in sorted(best)]
    sub = pd.concat(parts, ignore_index=True)
    sub.insert(0, "id", range(len(sub)))
    sub.to_csv(out, index=False)
    total = sum(best[n][0] / math.sqrt(n) - 1 for n in best)
    total += sum(math.ceil(math.sqrt(n)) / math.sqrt(n) - 1 for n in range(1, 1001) if n not in best)
    print(f"expected score: {total:.6f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
