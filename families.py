"""Closed-form packing families, written as candidate JSON for merge.py.

Goebel squares: for a, b >= 1 with a - 1 < b/sqrt2 < a + 1, n = 2a(a+1) + b^2 squares
fit in side s = a + 1 + b/sqrt2: a b x b block tilted 45 degrees in the center and
four corner staircases, the corner cell (i, j) kept when i + j <= a - 1.

    python families.py cands/goebel_squares.json
"""

import json
import sys

import mpmath

mpmath.mp.dps = 50
MAX_N = 1000


def goebel_square(a, b):
    r2 = mpmath.sqrt(2)
    s = a + 1 + b / r2
    c = s / 2
    half = mpmath.mpf(1) / 2
    sq = []
    for i in range(a):
        for j in range(a - i):
            for fx, fy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                x = i + half if fx == 0 else s - i - half
                y = j + half if fy == 0 else s - j - half
                sq.append((x, y, mpmath.mpf(0)))
    # b x b block rotated 45 degrees about the center: local cell centers
    # (u, v) in [-b/2, b/2], mapped by rotation.
    for p in range(b):
        for q in range(b):
            u = p - (b - 1) * half
            v = q - (b - 1) * half
            sq.append((c + (u - v) / r2, c + (u + v) / r2, mpmath.pi / 4))
    return s, sq


def main(out):
    res = {}
    r2 = mpmath.sqrt(2)
    for a in range(0, 40):
        for b in range(1, 50):
            if not (a - 1 < b / r2 < a + 1):
                continue
            n = 2 * a * (a + 1) + b * b
            if n > MAX_N:
                continue
            s, sq = goebel_square(a, b)
            assert len(sq) == n
            if n not in res or s < mpmath.mpf(res[n]["side"]):
                res[n] = {"side": mpmath.nstr(s, 40),
                          "squares": [[mpmath.nstr(v, 40) for v in t] for t in sq]}
    json.dump(res, open(out, "w"))
    print(f"{len(res)} Goebel-square packings")


if __name__ == "__main__":
    main(sys.argv[1])
