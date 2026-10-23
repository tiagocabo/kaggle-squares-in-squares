"""Local feasibility check + score for a submission CSV (float SAT, strict gaps)."""

import math
import sys

import numpy as np
import pandas as pd

GAP = 1e-14  # every candidate pair must be separated by more than this


def corners(x, y, a):
    c, s = np.cos(a), np.sin(a)
    d = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)]) / 2
    dx = d[:, 0][None] * c[:, None] - d[:, 1][None] * s[:, None]
    dy = d[:, 0][None] * s[:, None] + d[:, 1][None] * c[:, None]
    return np.stack([x[:, None] + dx, y[:, None] + dy], -1)  # (n, 4, 2)


def check(n, g):
    if len(g) != n or set(g.square_id) != set(range(n)):
        return None, "bad ids/count"
    x, y, a = (g[c].astype(float).to_numpy() for c in ("x", "y", "angle"))
    C = corners(x, y, a)
    L = max(C[..., 0].max() - C[..., 0].min(), C[..., 1].max() - C[..., 1].min())
    lo, hi = C.min(1), C.max(1)
    worst = math.inf
    for i in range(n):
        m = np.all(lo[i + 1:] <= hi[i] + 1e-9, 1) & np.all(hi[i + 1:] >= lo[i] - 1e-9, 1)
        for j in np.nonzero(m)[0] + i + 1:
            sep = -math.inf
            for ang in (a[i], a[j]):
                for ax in ((math.cos(ang), math.sin(ang)), (-math.sin(ang), math.cos(ang))):
                    p1, p2 = C[i] @ ax, C[j] @ ax
                    sep = max(sep, max(p2.min() - p1.max(), p1.min() - p2.max()))
            worst = min(worst, sep)
            if sep <= GAP:
                return None, f"pair {i},{j} sep={sep:.3e}"
    return L, f"ok (min sep {worst:.2e})"


def main(path):
    df = pd.read_csv(path, dtype={"x": str, "y": str, "angle": str})
    total, bad = 0.0, 0
    for n in range(1, 1001):
        g = df[df.n == n]
        L, msg = check(n, g) if len(g) else (None, "missing")
        if L is None:
            bad += 1
            print(f"n={n}: {msg}")
            L = math.ceil(math.sqrt(n))
        total += L / math.sqrt(n) - 1
    print(f"infeasible: {bad}  score: {total:.6f}")


if __name__ == "__main__":
    main(sys.argv[1])
