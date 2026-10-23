"""Convert exact rational certificates (franciscouzo/square-packing) to candidate JSON.

Cert format: a comment line, then "n side", then n lines "x y t" with x, y the
square center and t = tan(theta/2), all exact rationals.  Output keeps n up to
MAX_SOURCE_N so merge.py can delete squares from slightly larger witnesses.

    python parse_certs.py refs/square-packing/certificates/best cands/couzo.json
"""

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import mpmath

mpmath.mp.dps = 60
MAX_SOURCE_N = 1100


def to_mp(s):
    f = Fraction(s)
    return mpmath.mpf(f.numerator) / f.denominator


def parse(path):
    lines = [ln.split() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    n = int(lines[0][0])
    side = to_mp(lines[0][1])
    sq = []
    for x, y, t in lines[1:n + 1]:
        sq.append((to_mp(x), to_mp(y), 2 * mpmath.atan(to_mp(t))))
    if len(sq) != n:
        raise ValueError(f"{path}: {len(sq)} squares for n={n}")
    return n, side, sq


def main(cert_dir, out, max_n=MAX_SOURCE_N, min_n=0):
    res = {}
    for p in sorted(Path(cert_dir).glob("n*.cert")):
        m = re.match(r"n(\d+)$", p.stem)
        if not m or not (min_n <= int(m.group(1)) <= max_n):
            continue
        n, side, sq = parse(p)
        res[n] = {"side": mpmath.nstr(side, 40),
                  "squares": [[mpmath.nstr(v, 45) for v in s] for s in sq]}
    json.dump(res, open(out, "w"))
    print(f"{len(res)} certificates converted")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:5]))
