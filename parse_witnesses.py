"""Convert jlevy/squares witness YAMLs (packing/witnesses/**) to candidate JSON.

Supported representations (others are skipped):
  * corners, rational        -> center = mean of corners, angle from first edge
  * center-basis, rational   -> center, angle = atan2(basis)
  * center-angle, decimal    -> center, angle in degrees or radians
merge.py re-normalizes position and re-checks every packing, so origin
conventions don't matter.

    python parse_witnesses.py refs/jlevy_squares/packing/witnesses cands/jlevy.json
"""

import gzip
import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath
import yaml

mpmath.mp.dps = 60


def num(s):
    s = str(s).strip()
    if "/" in s or s.lstrip("-").isdigit():
        f = Fraction(s)
        return mpmath.mpf(f.numerator) / f.denominator
    return mpmath.mpf(s)


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return yaml.safe_load(f)


def squares_of(w):
    rep = w.get("representation")
    kind = (w.get("scalar") or {}).get("kind")
    unit = (w.get("coordinates") or {}).get("angle_unit")
    out = []
    for sq in w["squares"]:
        if rep == "corners" and kind == "rational":
            cs = [(num(x), num(y)) for x, y in sq["corners"]]
            cx = sum(c[0] for c in cs) / 4
            cy = sum(c[1] for c in cs) / 4
            a = mpmath.atan2(cs[1][1] - cs[0][1], cs[1][0] - cs[0][0])
        elif rep == "center-basis" and kind == "rational":
            cx, cy = (num(v) for v in sq["center"])
            bx, by = (num(v) for v in sq["basis"])
            a = mpmath.atan2(by, bx)
        elif rep == "center-angle" and kind == "decimal":
            cx, cy = num(sq["center"][0]), num(sq["center"][1])
            a = num(sq["angle"])
            if unit == "degrees":
                a = a * mpmath.pi / 180
        else:
            return None
        out.append((cx, cy, a))
    return out


def main(wdir, out):
    res, skipped = {}, 0
    for p in sorted(Path(wdir).rglob("*.yaml*")):
        if "schema" in p.name:
            continue
        doc = load(p)
        w = (doc or {}).get("witness")
        if not w or "squares" not in w:
            skipped += 1
            continue
        sq = squares_of(w)
        n = int(w["n"])
        if sq is None or len(sq) != n:
            skipped += 1
            continue
        key = f"{n}"
        # keep every distinct witness: merge picks the best valid one
        while key in res:
            key += "_"
        res[key] = {"file": str(p.relative_to(wdir)),
                    "squares": [[mpmath.nstr(v, 45) for v in s] for s in sq]}
    json.dump(res, open(out, "w"))
    print(f"{len(res)} witnesses converted, {skipped} skipped")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
