"""First submission for Kaggle "Squares in Squares".

For each n, take the better of:
  * the trivial grid (side ceil(sqrt(n)))
  * a Goebel-style diagonal-strip packing: a band of b rows of 45-degree squares
    along a diagonal y - x in [lo, lo + b*sqrt2], with axis-aligned staircases of
    unit squares filling the top-left and bottom-right corners.

Contacts are exact, so the final coordinates are scaled by (1 + EPS) around the
origin; that turns every contact into a strictly positive gap the metric's
interval-arithmetic check can certify.
"""

import math
import sys

import mpmath
import numpy as np

mpmath.mp.dps = 40
SQ2 = math.sqrt(2.0)
EPS = mpmath.mpf("1e-12")
TOL = 1e-12
MAX_N = 1000
MAX_B = 8
SLACK_L = 3e-9


def staircase_count(L, t):
    """Unit squares [i,i+1]x[L-j-1,L-j], i,j in [0,floor(L)-1], with i+j <= floor(t)."""
    K = int(math.floor(L + TOL))
    T = math.floor(t + TOL)
    if K <= 0 or T < 0:
        return 0
    i = np.arange(K)
    return int(np.clip(T - i + 1, 0, K).sum())


def row_count(L, p):
    """45-degree squares in a row at perpendicular offset p = (x - y)/sqrt2."""
    span = L * SQ2 - 2 - 2 * abs(p)
    return int(math.floor(span + TOL)) + 1 if span >= -TOL else 0


def strip_rows(lo, b):
    hi = lo + b * SQ2
    return [-hi / SQ2 + 0.5 + r for r in range(b)]


def strip_count(L, lo, b):
    hi = lo + b * SQ2
    n = staircase_count(L, L - 2 - hi) + staircase_count(L, L - 2 + lo)
    return n + sum(row_count(L, p) for p in strip_rows(lo, b))


def lo_candidates(L, b):
    """Strip positions where some count is about to drop: optimum is at one of these."""
    c = set()
    for m in range(-1, int(L) + 2):
        c.add(m - L + 2)                    # lower staircase tight
        c.add(L - 2 - m - b * SQ2)          # upper staircase tight
    for m in range(0, int(L * SQ2) + 2):
        half = (L * SQ2 - 2 - m) / 2        # row count jumps at |p| = half
        if half < 0:
            continue
        for r in range(b):
            for p in (half, -half):
                # p = -(lo + b*sqrt2)/sqrt2 + 0.5 + r  ->  solve for lo
                c.add((0.5 + r - p) * SQ2 - b * SQ2)
    return [lo for lo in c if -L - 1 <= lo <= L + 1]


def best_strip(L, b):
    best = (0, None)
    for lo in lo_candidates(L, b):
        k = strip_count(L, lo, b)
        if k > best[0]:
            best = (k, lo)
    return best


def min_side(n, b, lo_bound, hi_bound):
    """Bisect the smallest L (within [lo_bound, hi_bound]) whose strip packing holds n."""
    if best_strip(hi_bound, b)[0] < n:
        return None
    a, z = lo_bound, hi_bound
    for _ in range(45):
        m = (a + z) / 2
        if best_strip(m, b)[0] >= n:
            z = m
        else:
            a = m
    return z


def strip_squares(L, lo, b):
    """Return list of (x, y, angle) as mpf, exact construction at side L.

    No tolerance here: callers pass (L, lo) with ~1e-9 slack on every constraint,
    so floors are unambiguous and every admitted square really fits.
    """
    L = mpmath.mpf(L)
    lo = mpmath.mpf(lo)
    s2 = mpmath.sqrt(2)
    hi = lo + b * s2
    out = []
    K = int(mpmath.floor(L))
    T_up = mpmath.floor(L - 2 - hi)
    T_dn = mpmath.floor(L - 2 + lo)
    for i in range(K):
        for j in range(K):
            if i + j <= T_up:
                out.append((i + mpmath.mpf(0.5), L - j - mpmath.mpf(0.5), mpmath.mpf(0)))
            if i + j <= T_dn:
                out.append((L - i - mpmath.mpf(0.5), j + mpmath.mpf(0.5), mpmath.mpf(0)))
    for r in range(b):
        p = -hi / s2 + mpmath.mpf(0.5) + r
        span = L * s2 - 2 - 2 * abs(p)
        if span < 0:
            continue
        cnt = int(mpmath.floor(span)) + 1
        for k in range(cnt):
            s = 1 + abs(p) + k
            out.append(((s + p) / s2, (s - p) / s2, mpmath.pi / 4))
    return out


def grid_squares(n):
    k = math.ceil(math.sqrt(n))
    return [(mpmath.mpf(i % k) + mpmath.mpf(0.5), mpmath.mpf(i // k) + mpmath.mpf(0.5), mpmath.mpf(0))
            for i in range(n)]


def side_of(squares):
    xs, ys = [], []
    for x, y, a in squares:
        c, s = mpmath.cos(a), mpmath.sin(a)
        for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            xs.append(x + (dx * c - dy * s) / 2)
            ys.append(y + (dx * s + dy * c) / 2)
    return max(max(xs) - min(xs), max(ys) - min(ys))


def fmt(v):
    return mpmath.nstr(v, 30, strip_zeros=False, min_fixed=-10**9, max_fixed=10**9)


def main(out_path):
    rows = []
    summary = []
    for n in range(1, MAX_N + 1):
        grid_L = math.ceil(math.sqrt(n))
        best = (float(grid_L), "grid", None)
        for b in range(1, MAX_B + 1):
            L = min_side(n, b, math.sqrt(n), grid_L)
            if L is not None and L < best[0] - 1e-9:
                best = (L, "strip", b)
        if best[1] == "grid":
            sq = grid_squares(n)
        else:
            L, _, b = best
            _, lo = best_strip(L, b)
            # Slack: L grows 3e-9 and the strip slides 1.5e-9, which loosens the
            # staircase, row and wall constraints by >= 1e-9 each.
            sq = strip_squares(L + SLACK_L, lo + SLACK_L / 2, b)
            assert len(sq) >= n, (n, len(sq))
            sq = sq[:n]
        sq = [(x * (1 + EPS), y * (1 + EPS), a) for x, y, a in sq]
        L_meas = float(side_of(sq))
        summary.append((n, best[1], best[2], L_meas, grid_L))
        for sid, (x, y, a) in enumerate(sq):
            rows.append((n, sid, fmt(x), fmt(y), fmt(a)))
        if n % 100 == 0:
            print(f"n={n}", file=sys.stderr)

    with open(out_path, "w") as f:
        f.write("id,n,square_id,x,y,angle\n")
        for idx, (n, sid, x, y, a) in enumerate(rows):
            f.write(f"{idx},{n},{sid},{x},{y},{a}\n")

    improved = [s for s in summary if s[1] == "strip"]
    total = sum(L / math.sqrt(n) - 1 for n, _, _, L, _ in summary)
    grid_total = sum(g / math.sqrt(n) - 1 for n, _, _, _, g in summary)
    print(f"improved puzzles: {len(improved)}")
    print(f"expected score: {total:.6f}  (grid baseline {grid_total:.6f})")
    for s in improved[:15]:
        print(s)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "submission.csv")
