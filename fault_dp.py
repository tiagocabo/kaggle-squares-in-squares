"""Fault-line gadget composition (v3).

Idea (after Chris Deotte's "Multi-Gadget Fault DP" notebook): split a K x K grid of
unit cells along a monotone staircase from top-left to bottom-right.  Cells below /
left of it stay on the integer lattice ("00"); cells above / right are shifted by
(+f, +f) ("11").  Side S = K + f holds K^2 squares.  A *gadget* is a crop of a
known packing whose squares fit in [0, w+f_g] x [0, h+f_g] while numbering
w*h + d (d >= 1).  Gadgets sit on the staircase, replacing their w x h footprint;
chained top-left -> bottom-right they each see the same (+f, +f) jump, so

    N = K^2 + sum(d_m),   S = K + max(f_m).

Consecutive gadgets are separated by a buffer column (mode X), a buffer row
(mode Y), or nothing (mode 0, allowed when the first gadget's bottom-right tip or
the second's top-left tip is empty).

Pipeline:
  1. donors: every packing in a submission CSV + candidate JSONs (any n)
  2. gadgets: for each donor, each of the 8 symmetries and each (w, h) <= MAX_WH,
     r_i = max(0, xhi_i - w, yhi_i - h); sorting r gives the smallest f for each d
  3. DP over chains, per (W used, H used, total d, last-tip-free) -> min max f
  4. build each composition, verify strictly, write candidate JSON for merge.py

    python fault_dp.py submission_v2.csv cands/fault.json --donors cands/couzo_big.json
"""

import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from check_submission import check

MAX_WH = 31
MAX_D = 64
F_MAX = 0.999
TIP = 1.0  # conservative tip box size for mode-0 compatibility


def half_ext(a):
    return (np.abs(np.cos(a)) + np.abs(np.sin(a))) / 2


def normalize(p):
    p = p.copy()
    h = half_ext(p[:, 2])
    p[:, 0] -= (p[:, 0] - h).min()
    p[:, 1] -= (p[:, 1] - h).min()
    return p


def symmetries(p):
    """The 8 dihedral images of a packing (angles adjusted), each normalized."""
    out = []
    for tr in (False, True):
        q = p.copy()
        if tr:  # reflect across y = x
            q = np.stack([p[:, 1], p[:, 0], np.pi / 2 - p[:, 2]], 1)
        for fx in (False, True):
            for fy in (False, True):
                r = q.copy()
                if fx:
                    r[:, 0] = -r[:, 0]
                    r[:, 2] = np.pi - r[:, 2]
                if fy:
                    r[:, 1] = -r[:, 1]
                    r[:, 2] = -r[:, 2]
                out.append(normalize(r))
    return out


def extract(args):
    """Best gadget per (w, h, d) for one donor: list of (w, h, d, f, tl, br, donor, sym)."""
    did, p = args
    res = {}
    for s, q in enumerate(symmetries(p)):
        he = half_ext(q[:, 2])
        xlo, xhi, ylo, yhi = q[:, 0] - he, q[:, 0] + he, q[:, 1] - he, q[:, 1] + he
        n = len(q)
        hs = np.arange(1, MAX_WH + 1)
        for w in range(1, MAX_WH + 1):
            # r[i, h-1] = how far square i sticks out of the w x h crop
            r = np.maximum(np.maximum(xhi - w, 0)[:, None], (yhi[:, None] - hs[None, :]))
            r = np.maximum(r, 0)
            order = np.argsort(r, axis=0, kind="stable")
            rs = np.take_along_axis(r, order, axis=0)
            for h in range(1, MAX_WH + 1):
                base = w * h
                if base + 1 > n:
                    continue
                col = rs[:, h - 1]
                for d in range(1, min(MAX_D, n - base) + 1):
                    f = col[base + d - 1]
                    if f >= F_MAX:
                        break
                    if base + d < n and col[base + d] <= f + 1e-12:
                        continue  # same f admits more squares: the larger d dominates
                    key = (w, h, d)
                    if key in res and res[key][0] <= f:
                        continue
                    idx = order[: base + d, h - 1]
                    # tips: does any gadget square poke into the corner boxes?
                    br = not np.any((xhi[idx] > w + 1e-9) & (ylo[idx] < TIP))
                    tl = not np.any((yhi[idx] > h + 1e-9) & (xlo[idx] < TIP))
                    res[key] = (float(f), tl, br, did, s)
    return [(w, h, d, *v) for (w, h, d), v in res.items()]


def gadget_squares(p, sym, w, h, d):
    q = symmetries(p)[sym]
    he = half_ext(q[:, 2])
    r = np.maximum(np.maximum(q[:, 0] + he - w, q[:, 1] + he - h), 0)
    idx = np.argsort(r, kind="stable")[: w * h + d]
    return q[idx]


def load_donors(base_csv, extra_json):
    donors = []
    df = pd.read_csv(base_csv)
    for n, g in df.groupby("n"):
        g = g.sort_values("square_id")
        donors.append(normalize(g[["x", "y", "angle"]].to_numpy(float)))
    for path in extra_json:
        for key, rec in json.load(open(path)).items():
            donors.append(normalize(np.array(rec["squares"], dtype=float)))
    return donors


def build_groups(gadgets):
    """Group by (w, h, tl, br): fvec[d] = min f, plus the gadget achieving it."""
    groups = {}
    for w, h, d, f, tl, br, did, s in gadgets:
        key = (w, h, int(tl), int(br))
        fv, src = groups.setdefault(key, (np.full(MAX_D + 1, np.inf), {}))
        if f < fv[d]:
            fv[d] = f
            src[d] = (did, s)
    return groups


def run_dp(groups, rounds):
    """dp[r][b][W, H, D] = min max-f over chains of r+1 gadgets; b = last br_free."""
    shape = (MAX_WH + 2, MAX_WH + 2, MAX_D + 1)
    first = [np.full(shape, np.inf), np.full(shape, np.inf)]
    for (w, h, tl, br), (fv, _) in groups.items():
        first[br][w, h, :] = np.minimum(first[br][w, h, :], fv)
    layers = [first]
    for _ in range(rounds):
        prev = layers[-1]
        nxt = [np.full(shape, np.inf), np.full(shape, np.inf)]
        for (w, h, tl, br), (fv, _) in groups.items():
            ds = np.nonzero(np.isfinite(fv))[0]
            for b in (0, 1):
                src = prev[b]
                if not np.isfinite(src).any():
                    continue
                modes = [(1, 0), (0, 1)] + ([(0, 0)] if (b or tl) else [])
                for bx, by in modes:
                    dw, dh = w + bx, h + by
                    if dw > MAX_WH or dh > MAX_WH:
                        continue
                    a = src[: MAX_WH + 2 - dw, : MAX_WH + 2 - dh, :]
                    for d in ds:
                        tgt = nxt[br][dw:, dh:, d:]
                        cand = np.maximum(a[:, :, : MAX_D + 1 - d], fv[d])
                        np.minimum(tgt, cand, out=tgt)
        layers.append(nxt)
    return layers


def backtrack(layers, groups, r, b, W, H, D, f):
    """Recover one chain (list of (w, h, d, (donor, sym), mode)) reaching layers[r][b][W,H,D] <= f."""
    if r == 0:
        for (w, h, tl, br), (fv, src) in groups.items():
            if br == b and w == W and h == H and fv[D] <= f + 1e-15:
                return [(w, h, D, src[D], None)]
        return None
    prev = layers[r - 1]
    for (w, h, tl, br), (fv, src) in groups.items():
        if br != b:
            continue
        for d in np.nonzero(fv <= f + 1e-15)[0]:
            if d > D:
                continue
            for pb in (0, 1):
                modes = [("X", 1, 0), ("Y", 0, 1)] + ([("0", 0, 0)] if (pb or tl) else [])
                for name, bx, by in modes:
                    W0, H0 = W - w - bx, H - h - by
                    if W0 < 1 or H0 < 1 or prev[pb][W0, H0, D - d] > f + 1e-15:
                        continue
                    chain = backtrack(layers, groups, r - 1, pb, W0, H0, D - d, f)
                    if chain is not None:
                        return chain + [(w, h, d, src[d], name)]
    return None


def compose(chain, donors, K):
    """Place the chain on the staircase of a K x K grid; return (n, squares)."""
    parts = [gadget_squares(donors[did], s, w, h, d) for w, h, d, (did, s), _ in chain]
    f = max(max(0.0, (p[:, 0] + half_ext(p[:, 2])).max() - w, (p[:, 1] + half_ext(p[:, 2])).max() - h)
            for p, (w, h, *_rest) in zip(parts, chain))
    pos, ox, oy = [], 0, K
    for i, (w, h, d, _, mode) in enumerate(chain):
        if i == 0:
            ox, oy = 0, K - h
        else:
            pw = chain[i - 1][0]
            ox = ox + pw + (1 if mode == "X" else 0)
            oy = oy - h - (1 if mode == "Y" else 0)
        pos.append((ox, oy))
    # column thresholds: rows below are lattice "00", rows above are shifted "11"
    t = np.empty(K, dtype=int)
    gcell = np.zeros((K, K), dtype=bool)
    col = 0
    for i, ((ox, oy), (w, h, *_)) in enumerate(zip(pos, chain)):
        if ox > col:  # mode-X buffer column(s): threshold of the previous gadget
            t[col:ox] = pos[i - 1][1]
        t[ox:ox + w] = oy
        gcell[ox:ox + w, oy:oy + h] = True
        col = ox + w
    t[col:] = pos[-1][1]
    sq = []
    for i in range(K):
        for j in range(K):
            if gcell[i, j]:
                continue
            if j < t[i]:
                sq.append((i + 0.5, j + 0.5, 0.0))
            else:
                sq.append((i + 0.5 + f, j + 0.5 + f, 0.0))
    for p, (ox, oy) in zip(parts, pos):
        q = p.copy()
        q[:, 0] += ox
        q[:, 1] += oy
        sq.extend(map(tuple, q))
    return np.array(sq), K + f


def frame(n, sq):
    return pd.DataFrame({"n": n, "square_id": range(n), "x": [repr(float(v)) for v in sq[:, 0]],
                         "y": [repr(float(v)) for v in sq[:, 1]],
                         "angle": [repr(float(v)) for v in sq[:, 2]]})


def current_sides(base_csv):
    df = pd.read_csv(base_csv)
    out = {}
    for n, g in df.groupby("n"):
        p = g[["x", "y", "angle"]].to_numpy(float)
        he = half_ext(p[:, 2])
        out[n] = max(np.ptp(np.r_[p[:, 0] - he, p[:, 0] + he]), np.ptp(np.r_[p[:, 1] - he, p[:, 1] + he]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--donors", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--kmin", type=int, default=18)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()

    t0 = time.time()
    donors = load_donors(a.base, a.donors)
    with Pool(a.workers) as pool:
        gadgets = [g for part in pool.imap_unordered(extract, enumerate(donors), chunksize=4) for g in part]
    print(f"{len(donors)} donors -> {len(gadgets)} gadget rows ({time.time() - t0:.0f}s)", flush=True)
    groups = build_groups(gadgets)
    layers = run_dp(groups, a.rounds)
    print(f"DP done ({time.time() - t0:.0f}s)", flush=True)

    best = current_sides(a.base)
    res, gain = {}, 0.0
    for K in range(a.kmin, 32):
        for D in range(1, 2 * K + 2):
            n = K * K + D
            if n > 1000:
                break
            # best f over all chain lengths, flags, and footprints that fit in K
            cand = []
            for r, layer in enumerate(layers):
                for b in (0, 1):
                    sub = layer[b][1:K + 1, 1:K + 1, D]
                    if np.isfinite(sub).any():
                        W, H = np.unravel_index(np.argmin(sub), sub.shape)
                        cand.append((sub[W, H], r, b, W + 1, H + 1))
            if not cand:
                continue
            f, r, b, W, H = min(cand)
            # best is non-decreasing in n (drop sweep), so beating best[n] is the test
            if K + f >= best.get(n, 1e9) - 1e-9:
                continue
            chain = backtrack(layers, groups, r, b, W, H, D, f)
            if chain is None:
                print(f"K={K} D={D}: backtrack failed")
                continue
            sq, S = compose(chain, donors, K)
            if len(sq) != n:
                print(f"K={K} D={D}: built {len(sq)} squares, expected {n}")
                continue
            sq = normalize(sq)
            sq[:, :2] *= 1 + 1e-12  # exact contacts -> certified gaps, as in merge.py
            L, msg = check(n, frame(n, sq))
            if L is None:
                print(f"K={K} D={D} chain={[(c[0], c[1], c[2], c[4]) for c in chain]}: invalid ({msg})")
                continue
            if L < best.get(n, 1e9) - 1e-9:
                gain += (best[n] - L) / math.sqrt(n)
            res[str(n)] = {"side": repr(L), "chain": [(int(c[0]), int(c[1]), int(c[2]), c[4]) for c in chain],
                           "squares": sq.tolist()}
            print(f"n={n} (K={K}, D={D}): {best.get(n):.6f} -> {L:.6f}  chain={[(c[0], c[1], c[2], c[4]) for c in chain]}", flush=True)
    json.dump(res, open(a.out, "w"))
    print(f"{len(res)} compositions written; direct gain {gain:.6f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
