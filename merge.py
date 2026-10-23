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
MAX_N = 1000
MAX_SOURCE_N = 1100  # larger witnesses only feed the drop-a-square pass
MAX_PASSES = 4


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


def side_mp(sq):
    xs, ys = [], []
    for x, y, a in sq:
        r = abs(mpmath.cos(a)) / 2 + abs(mpmath.sin(a)) / 2
        xs += [x - r, x + r]
        ys += [y - r, y + r]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def improves(best, n, L):
    return n <= MAX_SOURCE_N and (n not in best or L < best[n][0] - 1e-12)


def drop_pass(best):
    """s(n) <= s(n+1): dropping a square from a valid packing keeps it valid."""
    won = 0
    for n in range(MAX_SOURCE_N - 1, 0, -1):
        if n + 1 in best and improves(best, n, best[n + 1][0]):
            g = best[n + 1][1].sort_values("square_id").iloc[:n].copy()
            g["n"] = n
            L, msg = check(n, g)
            if L is not None and improves(best, n, L):
                best[n] = (L, g, f"drop<-{n + 1}")
                won += 1
    print(f"drop-a-square sweep: improved {won} puzzles")
    return won


def extend_pass(best):
    """Add m "L"s: grow side s to s+m, fill the new L strip with grid squares.

    The column [s, s+m] x [0, s+m] takes m*floor(s+m) squares, the row
    [0, s] x [s, s+m] takes m*floor(s).
    """
    won = 0
    for n in range(1, MAX_SOURCE_N):
        if n not in best:
            continue
        s_est = best[n][0]
        sq = None
        for m in range(1, 8):
            n2 = n + m * (math.floor(s_est + m + 1e-9) + math.floor(s_est + 1e-9))
            if n2 > MAX_SOURCE_N:
                break
            if not improves(best, n2, s_est + m + 1e-9):
                continue
            if sq is None:
                g = best[n][1].sort_values("square_id")
                sq = normalize(list(zip(g.x, g.y, g.angle)))
                s = side_mp(sq)
            half = mpmath.mpf(1) / 2
            new = list(sq)
            new += [(s + i + half, j + half, mpmath.mpf(0))
                    for i in range(m) for j in range(int(mpmath.floor(s + m)))]
            new += [(i + half, s + j + half, mpmath.mpf(0))
                    for i in range(int(mpmath.floor(s))) for j in range(m)]
            new = new[:n2] if len(new) >= n2 else None
            if new is None:
                continue
            g2 = frame(n2, normalize(new))
            L, msg = check(n2, g2)
            if L is not None and improves(best, n2, L):
                best[n2] = (L, g2, f"extend{m}<-{n}")
                won += 1
    print(f"L-extension pass: improved {won} puzzles")
    return won


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
            if n > MAX_SOURCE_N or len(rec["squares"]) != n:
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
    for _ in range(MAX_PASSES):
        changed = extend_pass(best) + drop_pass(best)
        if not changed:
            break
    best = {n: v for n, v in best.items() if n <= MAX_N}
    parts = [best[n][1] for n in sorted(best)]
    sub = pd.concat(parts, ignore_index=True)
    sub.insert(0, "id", range(len(sub)))
    sub.to_csv(out, index=False)
    total = sum(best[n][0] / math.sqrt(n) - 1 for n in best)
    total += sum(math.ceil(math.sqrt(n)) / math.sqrt(n) - 1 for n in range(1, 1001) if n not in best)
    print(f"expected score: {total:.6f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
