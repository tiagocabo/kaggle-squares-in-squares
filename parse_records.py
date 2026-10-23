"""Extract unit-square packings from David Ellsworth's record SVGs (refs/svg).

Each SVG draws the packing with nested transforms; entity values carry ~35
digits.  We walk the tree with a high-precision affine transform stack and turn
every filled shape into unit squares:
  * <rect> of integer w x h  -> w*h unit cells
  * filled <path> (rectilinear, integer local coords) -> unit cells inside it
  * <use> -> render the referenced element under the extra transform
Output: records.json  {n: {"file":..., "side": str, "squares": [[x,y,angle],...]}}
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mpmath

mpmath.mp.dps = 50
XL = "{http://www.w3.org/1999/xlink}href"
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def mpf(s):
    return mpmath.mpf(s.strip())


def mat_mul(A, B):
    a, b, c, d, e, f = A
    g, h, i, j, k, l = B
    return (a * g + c * h, b * g + d * h, a * i + c * j, b * i + d * j,
            a * k + c * l + e, b * k + d * l + f)


IDENT = tuple(mpmath.mpf(v) for v in (1, 0, 0, 1, 0, 0))


def parse_transform(t):
    M = IDENT
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", t or ""):
        v = [mpf(x) for x in re.findall(NUM, args)]
        if name == "translate":
            T = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            T = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            r = v[0] * mpmath.pi / 180
            c, s = mpmath.cos(r), mpmath.sin(r)
            T = (c, s, -s, c, 0, 0)
            if len(v) == 3:
                T = mat_mul(mat_mul((1, 0, 0, 1, v[1], v[2]), T), (1, 0, 0, 1, -v[1], -v[2]))
        elif name == "matrix":
            T = tuple(v)
        else:
            raise ValueError(f"transform {name}")
        M = mat_mul(M, tuple(mpmath.mpf(x) for x in T))
    return M


def path_polys(d):
    toks = re.findall(rf"[MmLlHhVvZz]|{NUM}", d)
    polys, cur, x, y, cmd, i = [], [], mpmath.mpf(0), mpmath.mpf(0), None, 0
    while i < len(toks):
        t = toks[i]
        if re.match(r"[A-Za-z]", t):
            cmd = t
            i += 1
            if cmd in "Zz":
                if cur:
                    polys.append(cur)
                cur = []
            continue
        if cmd in "Mm":
            nx, ny = mpf(toks[i]), mpf(toks[i + 1]); i += 2
            if cmd == "m":
                nx, ny = x + nx, y + ny
            if cur:
                polys.append(cur)
            cur = [(nx, ny)]
            x, y = nx, ny
            cmd = "L" if cmd == "M" else "l"
            continue
        if cmd in "Ll":
            nx, ny = mpf(toks[i]), mpf(toks[i + 1]); i += 2
            if cmd == "l":
                nx, ny = x + nx, y + ny
        elif cmd in "Hh":
            nx, ny = mpf(toks[i]) + (x if cmd == "h" else 0), y; i += 1
        elif cmd in "Vv":
            nx, ny = x, mpf(toks[i]) + (y if cmd == "v" else 0); i += 1
        else:
            raise ValueError(f"path cmd {cmd}")
        x, y = nx, ny
        cur.append((x, y))
    if cur:
        polys.append(cur)
    return polys


def inside(polys, px, py):
    """Even-odd point-in-polygon over all subpaths (implicitly closed)."""
    c = False
    for poly in polys:
        m = len(poly)
        for k in range(m):
            (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % m]
            if (y1 > py) != (y2 > py) and px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
                c = not c
    return c


def _uniq(vals):
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > 1e-30:
            out.append(v)
    return out


def _is_int(v):
    return abs(v - mpmath.nint(v)) < 1e-30


def cells_of_path(d):
    """Column-wise decomposition, falling back to row-wise when columns don't split."""
    polys = path_polys(d)
    try:
        return _cells(polys)
    except ValueError:
        swapped = [[(y, x) for x, y in poly] for poly in polys]
        return [(x, y) for y, x in _cells(swapped)]


def _cells(polys):
    """Decompose a filled rectilinear region into unit cells (lower-left corners).

    Columns between consecutive distinct x's must have integer width; inside a
    column the filled y-runs must have integer length.  Cells may sit at
    non-integer offsets (e.g. a column shifted up by some entity value).
    """
    pts = [p for poly in polys for p in poly]
    xs, ys = _uniq(p[0] for p in pts), _uniq(p[1] for p in pts)
    cells = []
    for x0, x1 in zip(xs, xs[1:]):
        xm = (x0 + x1) / 2
        runs = []
        for y0, y1 in zip(ys, ys[1:]):
            if inside(polys, xm, (y0 + y1) / 2):
                if runs and abs(runs[-1][1] - y0) < 1e-30:
                    runs[-1][1] = y1
                else:
                    runs.append([y0, y1])
        if runs and not _is_int(x1 - x0):
            raise ValueError("non-integer column width")
        for y0, y1 in runs:
            if not _is_int(y1 - y0):
                raise ValueError("non-integer run length")
            for i in range(int(mpmath.nint(x1 - x0))):
                for j in range(int(mpmath.nint(y1 - y0))):
                    cells.append((x0 + i, y0 + j))
    return cells


class Extractor:
    def __init__(self, root, side):
        self.side = side
        self.ids = {}
        for el in root.iter():
            if el.get("id") and el.get("id") not in self.ids:
                self.ids[el.get("id")] = el
        self.squares = []

    def emit_cell(self, M, i, j):
        a, b, c, d, e, f = M
        if abs(a * a + b * b - 1) > 1e-25 or abs(a * c + b * d) > 1e-25 or abs(c * c + d * d - 1) > 1e-25:
            raise ValueError("non-rigid transform on a square")
        cx, cy = i + mpmath.mpf(0.5), j + mpmath.mpf(0.5)
        self.squares.append((a * cx + c * cy + e, b * cx + d * cy + f, mpmath.atan2(b, a)))

    def walk(self, el, M, depth=0):
        tag = el.tag.split("}")[-1]
        if tag in ("defs", "script", "title", "desc", "style", "text", "metadata"):
            return
        style = (el.get("style") or "").replace(" ", "")
        if "fill:none" in style or "fill:white" in style or "display:none" in style:
            if tag != "g":
                return
        M = mat_mul(M, parse_transform(el.get("transform")))
        if tag == "use":
            ref = (el.get(XL) or el.get("href") or "").lstrip("#")
            if ref == "outer":
                return
            M = mat_mul(M, (1, 0, 0, 1, mpf(el.get("x", "0")), mpf(el.get("y", "0"))))
            self.walk(self.ids[ref], M, depth + 1)
        elif tag == "rect":
            w, h = mpf(el.get("width")), mpf(el.get("height"))
            if el.get("id") == "outer" or abs(w - self.side) < 1e-9:
                return
            x0, y0 = mpf(el.get("x", "0")), mpf(el.get("y", "0"))
            if w != mpmath.nint(w) or h != mpmath.nint(h):
                raise ValueError(f"rect {w}x{h}")
            M2 = mat_mul(M, (1, 0, 0, 1, x0, y0))
            for i in range(int(w)):
                for j in range(int(h)):
                    self.emit_cell(M2, i, j)
        elif tag == "path":
            for i, j in cells_of_path(el.get("d")):
                self.emit_cell(M, i, j)
        elif tag in ("g", "svg"):
            for ch in el:
                self.walk(ch, M, depth + 1)
        elif tag in ("line", "circle", "ellipse", "polyline", "polygon"):
            if tag in ("polygon",):
                raise ValueError("polygon")
        else:
            raise ValueError(f"tag {tag}")


def extract(path):
    root = ET.parse(path).getroot()
    if root.get("viewBox") is None:
        raise ValueError("no viewBox (multi-frame/animated variant)")
    vb = root.get("viewBox").split()
    side = mpf(vb[2])
    ex = Extractor(root, side)
    ex.walk(root, IDENT)
    return side, ex.squares


def main(svg_dir, out):
    res, errs = {}, []
    for p in sorted(Path(svg_dir).glob("square-*.svg")):
        m = re.match(r"square-(\d+)", p.stem)
        n = int(m.group(1))
        if n > 1000:
            continue
        try:
            side, sq = extract(p)
        except Exception as e:  # noqa: BLE001 - report and move on
            errs.append((p.name, repr(e)[:120]))
            continue
        if len(sq) != n:
            errs.append((p.name, f"got {len(sq)} squares"))
            continue
        prev = res.get(n)
        if prev is None or side < mpmath.mpf(prev["side"]):
            res[n] = {"file": p.name, "side": mpmath.nstr(side, 40),
                      "squares": [[mpmath.nstr(v, 40) for v in s] for s in sq]}
    json.dump(res, open(out, "w"))
    print(f"parsed {len(res)} n values; {len(errs)} problem files")
    for e in errs:
        print("  ", *e)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
