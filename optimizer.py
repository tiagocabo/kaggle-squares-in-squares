"""Local compaction optimizer: shrink the box around a valid packing.

For each n, repeatedly try side L - delta: scale the packing into the smaller box,
then minimize an overlap energy (squared SAT penetration depth per neighbouring
pair + squared wall violation) over (x, y, theta) with L-BFGS.  A step is accepted
only once the result passes the strict local check after a tiny legalising scale-up,
so every stored packing is already valid.  delta shrinks on failure.

    python optimizer.py submission_v2.csv cands/opt.json 101,102,... [--workers 10]
"""

import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from check_submission import check

torch.set_default_dtype(torch.float64)
NEIGHBOR_R = math.sqrt(2) + 0.3
OK_DEPTH = 1e-11


def energy(p, L, pairs):
    x, y, t = p[:, 0], p[:, 1], p[:, 2]
    c, s = torch.cos(t), torch.sin(t)
    h = (c.abs() + s.abs()) / 2
    wall = (torch.relu(h - x) ** 2 + torch.relu(x + h - L) ** 2
            + torch.relu(h - y) ** 2 + torch.relu(y + h - L) ** 2).sum()
    if pairs.shape[0] == 0:
        return wall, torch.zeros(())
    i, j = pairs[:, 0], pairs[:, 1]
    d = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=p.dtype) / 2
    cx = x[:, None] + d[None, :, 0] * c[:, None] - d[None, :, 1] * s[:, None]
    cy = y[:, None] + d[None, :, 0] * s[:, None] + d[None, :, 1] * c[:, None]
    axes = torch.stack([torch.stack([c[i], s[i]], -1), torch.stack([-s[i], c[i]], -1),
                        torch.stack([c[j], s[j]], -1), torch.stack([-s[j], c[j]], -1)], 1)
    pi = cx[i][:, None, :] * axes[..., 0:1] + cy[i][:, None, :] * axes[..., 1:2]
    pj = cx[j][:, None, :] * axes[..., 0:1] + cy[j][:, None, :] * axes[..., 1:2]
    ov = torch.minimum(pi.max(-1).values, pj.max(-1).values) - torch.maximum(pi.min(-1).values, pj.min(-1).values)
    depth = torch.relu(ov.min(-1).values)
    return wall + (depth ** 2).sum(), depth.max()


def neighbor_pairs(p):
    xy = p[:, :2]
    dist = torch.cdist(xy, xy)
    i, j = torch.nonzero(torch.triu(dist < NEIGHBOR_R, diagonal=1), as_tuple=True)
    return torch.stack([i, j], 1)


def relax(p0, L, iters=6):
    p = p0.clone().requires_grad_(True)
    for _ in range(iters):
        pairs = neighbor_pairs(p.detach())
        opt = torch.optim.LBFGS([p], lr=1, max_iter=200, tolerance_grad=1e-18,
                                tolerance_change=1e-22, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            e, _ = energy(p, L, pairs)
            e.backward()
            return e

        opt.step(closure)
        with torch.no_grad():
            e, dmax = energy(p, L, neighbor_pairs(p))
        if e.item() < OK_DEPTH ** 2:
            break
    return p.detach(), e.item()


def legalize(p, n):
    """Scale centers about the origin until the strict checker accepts; return (L, packing)."""
    q = p.numpy().copy()
    q[:, 0] -= (q[:, 0] - (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2]))) / 2).min()
    q[:, 1] -= (q[:, 1] - (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2]))) / 2).min()
    for eps in (1e-12, 1e-11, 1e-10, 1e-9):
        r = q.copy()
        r[:, :2] *= 1 + eps
        g = pd.DataFrame({"n": n, "square_id": range(n), "x": [repr(float(v)) for v in r[:, 0]],
                          "y": [repr(float(v)) for v in r[:, 1]], "angle": [repr(float(v)) for v in r[:, 2]]})
        L, _ = check(n, g)
        if L is not None:
            return L, r
    return None, None


def optimize(args):
    n, sq, L0, budget = args
    torch.set_num_threads(1)
    t_end = time.time() + budget
    p = torch.tensor(sq)
    best_L, best_p = L0, None
    delta = 2e-3 * L0
    gen = torch.Generator().manual_seed(n)
    while delta > 1e-7 * L0 and time.time() < t_end:
        L_try = best_L - delta
        ok = False
        # First a pure shrink, then basin-hopping kicks of growing size.
        for kick in (0.0, 0.02, 0.05, 0.1, 0.2):
            q = p.clone()
            q[:, :2] *= L_try / best_L
            if kick:
                q += kick * torch.randn(q.shape, generator=gen) * torch.tensor([1.0, 1.0, 0.5])
            q, e = relax(q.contiguous(), L_try)
            if e < OK_DEPTH ** 2:
                L_ok, r = legalize(q, n)
                if L_ok is not None and L_ok < best_L - 1e-10:
                    best_L, best_p, p = L_ok, r, torch.tensor(r)
                    ok = True
                    break
            if time.time() > t_end:
                break
        if not ok:
            delta /= 3
    return n, L0, best_L, (best_p.tolist() if best_p is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("ns")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--budget", type=float, default=120.0, help="seconds per puzzle")
    a = ap.parse_args()
    df = pd.read_csv(a.base)
    ns = [int(v) for v in a.ns.split(",")]
    jobs = []
    for n in ns:
        g = df[df.n == n].sort_values("square_id")
        sq = g[["x", "y", "angle"]].to_numpy(float)
        h = (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2]))) / 2
        L0 = max(np.ptp(np.r_[sq[:, 0] - h, sq[:, 0] + h]), np.ptp(np.r_[sq[:, 1] - h, sq[:, 1] + h]))
        jobs.append((n, sq.tolist(), L0, a.budget))
    try:
        res = json.load(open(a.out))
    except FileNotFoundError:
        res = {}
    gain = 0.0
    with Pool(a.workers) as pool:
        for n, L0, L, sq in pool.imap_unordered(optimize, jobs):
            if sq is not None:
                gain += (L0 - L) / math.sqrt(n)
                res[str(n)] = {"side": repr(L), "squares": sq}
                json.dump(res, open(a.out, "w"))
            print(f"n={n}: {L0:.9f} -> {L:.9f}  (score gain so far {gain:.6f})", flush=True)


if __name__ == "__main__":
    main()
