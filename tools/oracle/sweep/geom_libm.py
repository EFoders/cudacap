"""GEOM's control points under two float32 libms: glibc's and correctly rounded.

A stdlib-only re-statement of geom.for lines 48-160 (distance, bearing, and
the control-point latitude/longitude), every operation rounded to float32,
with the three transcendental calls (SIN, COS, ACOS) taken from either

* ``glibc`` -- the real ``sinf`` / ``cosf`` / ``acosf`` of ``libm.so.6``
  through ctypes (run it in WSL: this is what the gfortran oracle calls), or
* ``cr``    -- the double-precision value rounded to float32, which is what
  cudacap's host setup chain uses (``voacap_gpu.scalar.f32math``).

It shows how far a 1-ULP libm disagreement moves CLAT / CLONG for a given
path, i.e. whether a setup-chain difference is the libm floor amplified by
GEOM's conditioning rather than a port slip.

    wsl python3 tools/oracle/sweep/geom_libm.py TLAT TLON RLAT RLON NPSL
    wsl python3 tools/oracle/sweep/geom_libm.py --check DECKS DUMPS deck ...

``--check`` reads each deck's CIRCUIT card and its captured LUFFY entry state
and reports, per libm, how many of GCD / CLAT(k) / CLONG(k) match the capture
bit for bit (captured values are ES18.9 text, i.e. exact float32).
"""

import ctypes
import ctypes.util
import math
import struct
import sys


def f(x):
    return struct.unpack("f", struct.pack("f", x))[0]


D2R, PI, PI2, PIO2, RZ = (f(0.01745329251), f(3.1415926), f(6.283185307),
                          f(1.570796326), f(6370.0))
EPS = f(1.e-7)


def libms():
    out = {"cr": (lambda x: f(math.sin(x)), lambda x: f(math.cos(x)),
                  lambda x: f(math.acos(x)))}
    name = ctypes.util.find_library("m")
    if name and "libm" in name and sys.platform.startswith("linux"):
        m = ctypes.CDLL(name)
        fs = []
        for fn in ("sinf", "cosf", "acosf"):
            g = getattr(m, fn)
            g.restype, g.argtypes = ctypes.c_float, [ctypes.c_float]
            fs.append(g)
        out["glibc"] = tuple(fs)
    return out


def sign(a, b):
    return abs(a) if b >= 0 else -abs(a)


def clamp(q):
    return sign(f(1.0), q) if abs(q) > 1.0 else q


def geom(tlatd, tlongd, rlatd, rlongd, npsl, sin, cos, acos):
    tlat, tlong = f(f(tlatd) * D2R), f(f(tlongd) * D2R)
    rlat, rlong = f(f(rlatd) * D2R), f(f(rlongd) * D2R)
    dlong = f(tlong - rlong)
    if abs(dlong) > PI:
        dlong = f(dlong - sign(PI2, dlong))
    if npsl == 1:
        dlong = f(dlong - sign(PI2, dlong))
    q = f(f(sin(tlat) * sin(rlat)) + f(f(cos(tlat) * cos(rlat)) * cos(dlong)))
    gcd = acos(clamp(q))
    gcd = max(gcd, f(1e-6))
    if npsl == 1:
        gcd = f(PI2 - gcd)
    q = f(f(sin(rlat) - f(sin(tlat) * cos(gcd))) / f(cos(tlat) * sin(gcd)))
    btr = acos(clamp(q))
    if dlong > 0.0:
        btr = f(PI2 - btr)
    gcdkm = f(gcd * RZ)
    if gcdkm <= f(2000.01):
        rd = [f(gcd / 2.0)]
    elif gcdkm <= 4000.0:
        r1 = f(f(1000.0) / RZ)
        rd = [r1, f(gcd / 2.0), f(gcd - r1)]
    else:
        r1 = f(f(1000.0) / RZ)
        r2 = f(r1 + r1)
        rd = [r1, r2, f(gcd / 2.0), f(gcd - r2), f(gcd - r1)]
    pts = []
    for drf in rd:
        q = f(f(cos(drf) * sin(tlat)) + f(f(sin(drf) * cos(tlat)) * cos(btr)))
        rflt = f(PIO2 - acos(clamp(q)))
        if f(cos(rflt) - EPS) <= 0.0:
            rflg = tlong
        else:
            q = f(f(cos(drf) - f(sin(rflt) * sin(tlat)))
                  / f(cos(rflt) * cos(tlat)))
            rflg = acos(clamp(q))
            if drf >= PI:
                rflg = f(PI2 - rflg)
            rflg = f(tlong - sign(rflg, dlong))
            if abs(rflg) > PI:
                rflg = f(rflg - sign(PI2, rflg))
        pts.append((rflt, rflg))
    return gcd, gcdkm, btr, pts


def circuit_card(path):
    for line in open(path):
        if line.startswith("CIRCUIT"):
            sg = lambda v, h: -float(v) if h in "SW" else float(v)  # noqa: E731
            return (sg(line[10:15], line[15]), sg(line[16:25], line[25]),
                    sg(line[26:35], line[35]), sg(line[36:45], line[45]),
                    int(line[50:55]))
    raise LookupError(path)


def check(decks, dumps, names):
    import csv
    import os
    print("deck	libm	gcd	clat/clong exact	worst |d| rad")
    for name in names:
        tl, tg, rl, rg, npsl = circuit_card(os.path.join(decks, name + ".dat"))
        with open(os.path.join(dumps, name + "_in.csv")) as fh:
            r = csv.reader(fh)
            h = [x.strip() for x in next(r)]
            v = {k: f(float(x)) for k, x in zip(h, next(r))}
        for lib, (s, c, a) in libms().items():
            gcd, km, btr, pts = geom(tl, tg, rl, rg, npsl, s, c, a)
            n = ok = 0
            worst = 0.0
            for i, (la, lo) in enumerate(pts, 1):
                for got, key in ((la, f"clat{i:02d}"), (lo, f"clong{i:02d}")):
                    n += 1
                    ok += got == v[key]
                    worst = max(worst, abs(got - v[key]))
            print(f"{name}	{lib}	{'exact' if gcd == v['gcd'] else 'DIFF'}"
                  f"	{ok}/{n}	{worst:.3g}")


def main():
    if sys.argv[1] == "--check":
        return check(sys.argv[2], sys.argv[3], sys.argv[4:])
    tl, tg, rl, rg, npsl = (float(x) for x in sys.argv[1:6])
    for name, (s, c, a) in libms().items():
        gcd, km, btr, pts = geom(tl, tg, rl, rg, int(npsl), s, c, a)
        print(f"{name:6s} gcd {gcd!r} gcdkm {km!r} btr {btr!r}")
        for i, (la, lo) in enumerate(pts, 1):
            print(f"       clat{i:02d} {la!r:>22} clong{i:02d} {lo!r:>22}")


if __name__ == "__main__":
    main()
