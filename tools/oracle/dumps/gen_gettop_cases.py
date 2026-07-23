#!/usr/bin/env python3
"""Deterministic case grid for the GETTOP dump.

GETTOP finds how much of a long path the ray flies OVER the E layer -- the
night-day-night case, where a daylit middle reflects a frequency the dark ends
let through.  Five tests in a row can each zero the answer, and every one of
them has to be probed from both sides AND exactly on it.

This generator therefore emulates GETTOP's float32 arithmetic (struct-based
rounding, correctly-rounded COS/SQRT -- the same evaluation the port's
f32math helpers do) so it can SEARCH for the input that lands a boundary
expression on exactly zero, one float32 ULP at a time.  Where the emulation
and gfortran's libm disagree the oracle simply takes the other arm and the
value mismatch fails the test, which is the outcome we want.

What has to be reached:

* **FP1 vs FP2**, the middle-must-be-highest test, from both sides and
  EXACTLY EQUAL.  Equality is constructed exactly rather than searched: with
  all three E-layer heights identical the oblique divisor is the same float32
  number for all three areas, so equal FI gives bit-equal FPE.
* which of the two ends supplies FP1 -- area 1, area 3, and a tie.
* **FREQ against FP2 + 0.001 and against FP1 - 0.001**, each from both sides
  and exactly on.  The two slacks pull opposite ways, so the window is a
  kilohertz wider than the bracket at each end.
* **GMKM against 1000 km**, exactly on it and either side (GMKM is linear in
  DIKM, so DIKM is the search variable).
* **DIKM - GMKM against 1000 km**, likewise.
* **the sin^2 clamp at 0.9999**, which needs either DELOPT at zero with an
  E layer at ground level or a NEGATIVE height at DELOPT 3.  Neither is
  physical -- with the real DELOPT of 3 degrees and a real E layer near
  110 km, sin^2 is 0.964 and the clamp is unreachable -- so it is probed with
  artificial geometry and documented as dead for real input.  Without it a
  negative height would put SQRT of a negative number into the divisor.
* FI zero (a penetration frequency of exactly zero), and DIKM negative, which
  LNGPAT permits down to -1 km.

stdlib-only.  Usage: gen_gettop_cases.py <outdir>
"""

import math
import pathlib
import struct
import sys

SEED = 20260728

COLUMNS = ["delopt", "freq", "dikm", "hi1", "hi2", "hi3", "fi1", "fi2", "fi3"]

DEFAULTS = {
    "delopt": 3.0, "freq": 3.5, "dikm": 15000.0,
    "hi1": 110.0, "hi2": 110.0, "hi3": 110.0,
    "fi1": 0.6, "fi2": 0.8, "fi3": 0.6,
}


# --- float32 emulation, enough to mirror GETTOP exactly ------------------
def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bits(x):
    return struct.unpack("<i", struct.pack("<f", f32(x)))[0]


def _unbits(i):
    return struct.unpack("<f", struct.pack("<i", i))[0]


def step32(x, n):
    """x moved n float32 ULPs away from zero (x > 0 in every use here)."""
    b = _bits(x)
    return _unbits(b + n if b >= 0 else b - n)


D2R = f32(0.01745329251)
RZ = f32(6370.0)
F001 = f32(0.001)


def fpe_of(delopt, ht, fv):
    dele = f32(f32(delopt) * D2R)
    rcosd = f32(RZ * f32(math.cos(dele)))
    sphe = f32(rcosd / f32(RZ + f32(ht)))
    sphe = f32(sphe * sphe)
    sphe = min(sphe, f32(0.9999))
    return f32(f32(fv) / f32(math.sqrt(f32(f32(1.0) - sphe))))


def fps(delopt, hi, fi):
    fpe = [fpe_of(delopt, hi[j], fi[j]) for j in range(3)]
    return max(fpe[0], fpe[2]), fpe[1], min(fpe[0], fpe[2])


def raw_gmkm(delopt, freq, dikm, hi, fi):
    """The label-185 formula on its own, with neither 1000 km test applied."""
    fp1, fp2, fp3 = fps(delopt, hi, fi)
    span = f32(f32(f32(1.0) / f32(fp2 - fp1))
               + f32(f32(1.0) / f32(fp2 - fp3)))
    return f32(f32(f32(f32(0.5) * f32(dikm)) * f32(fp2 - f32(freq))) * span)


def search(start, expr, lo=-400, hi=400):
    """First ULP neighbour of `start` for which expr(x) is exactly 0.0."""
    for n in range(lo, hi + 1):
        x = step32(start, n)
        if expr(x) == 0.0:
            return x
    return None


def f(v) -> str:
    return f"{float(v):.9g}"


def case(**kw) -> str:
    unknown = set(kw) - set(COLUMNS)
    if unknown:
        raise KeyError(f"not a column: {sorted(unknown)}")
    d = dict(DEFAULTS)
    d.update(kw)
    return ",".join(f(d[c]) for c in COLUMNS)


def _fpe(delopt=3.0, ht=110.0, fv=1.0):
    return fpe_of(delopt, ht, fv)


def cases():
    rows = []
    flat = [110.0, 110.0, 110.0]

    # --- FP1 vs FP2: middle lower, EXACTLY EQUAL, middle higher -----------
    # equal heights => one shared divisor => equal FI gives bit-equal FPE
    for fi in ([0.9, 0.5, 0.4], [0.8, 0.8, 0.4], [0.8, 0.8, 0.8],
               [0.4, 0.8, 0.8], [0.4, 0.8, 0.6], [0.6, 0.8, 0.4],
               [0.4, 0.8, 0.4], [0.5, 0.5, 0.9], [0.9, 0.9, 0.5]):
        rows.append(case(fi1=fi[0], fi2=fi[1], fi3=fi[2]))
    # a tie between the two ENDS (FP1 == FP3), and each end winning
    rows.append(case(fi1=0.6, fi2=0.8, fi3=0.6))
    rows.append(case(fi1=0.7, fi2=0.8, fi3=0.5))
    rows.append(case(fi1=0.5, fi2=0.8, fi3=0.7))

    # --- FREQ against the two 0.001 MHz slacks, from both sides -----------
    # AT A PHYSICAL FREQUENCY THE BOUNDARY CANNOT BE HIT EXACTLY.  FREQ-FP2
    # is an exact subtraction of two nearby float32s, hence a multiple of
    # ulp(FP2); float32 0.001 is an ODD multiple of 2^-33, so the difference
    # can only equal it when ulp(FP2) is 2^-33 itself -- that is, |FP2| below
    # 2^-9 MHz.  Above that the arithmetic IF's zero arm is unreachable, and
    # since it shares a label with one of the others (235 and 235 for the
    # first test, 235 and 235 for the second) nothing observable rests on it.
    # Both sides are still pinned by stepping FREQ one ULP at a time.
    fp2 = _fpe(fv=0.8)
    fp1 = _fpe(fv=0.6)
    for n in (-40, -8, -1, 0, 1, 8, 40):
        rows.append(case(freq=step32(fp2 + 0.001, n)))
        rows.append(case(freq=step32(fp1 - 0.001, n)))
    # ... and here the zero arm IS reachable, at a kilohertz-scale FP.  The
    # search is two-dimensional: FP must also be an ODD multiple of 2^-33
    # (its own last mantissa bit set) or the difference lands on the even
    # grid and steps straight over float32 0.001, so FI is nudged too.
    d = _fpe(fv=1.0)                            # FPE per unit FI at HI=110

    def find_on(fi_seed, which, plus, guess):
        for m in range(0, 64):
            fi = list(fi_seed)
            fi[which] = step32(fi_seed[which], m)
            fp = fps(3.0, flat, fi)[0 if which == 0 else 1]
            hit = search(guess, (lambda p: lambda x: f32(f32(x - p) + F001))(fp)
                         if plus else
                         (lambda p: lambda x: f32(f32(x - p) - F001))(fp),
                         -4000, 4000)
            if hit is not None:
                return fi, hit
        return None, None

    tiny2, hit = find_on([0.0005 / d, 0.0011 / d, 0.0003 / d], 1, False,
                         0.0021)
    assert hit is not None, "no exact FREQ == FP2+0.001"
    for n in (-2, -1, 0, 1, 2):
        rows.append(case(freq=step32(hit, n), fi1=tiny2[0], fi2=tiny2[1],
                         fi3=tiny2[2]))
    tiny1, hit = find_on([0.0021 / d, 0.0030 / d, 0.0005 / d], 0, True,
                         0.0011)
    assert hit is not None, "no exact FREQ == FP1-0.001"
    for n in (-2, -1, 0, 1, 2):
        rows.append(case(freq=step32(hit, n), fi1=tiny1[0], fi2=tiny1[1],
                         fi3=tiny1[2]))
    # --- FREQ across the whole admissible window --------------------------
    for frac in (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98):
        rows.append(case(freq=fp1 + frac * (fp2 - fp1)))

    # --- the two 1000 km tests, exactly on and either side -----------------
    # Stepping DIKM alone cannot land GMKM on exactly 1000.0: GMKM is
    # A*SPAN with SPAN about 1.9, so the half-ULP window around 1000 is
    # NARROWER in A than A's own grid spacing, and roughly half of all
    # targets have no preimage.  Nudging FREQ does not help either -- SPAN
    # does not depend on it, so the window sits at a fixed place on a fixed
    # grid.  FI(2) is the second dimension that actually moves both.
    def find_dikm_on(want_left_over):
        for m in range(0, 512):
            fi = [0.6, step32(0.8, m), 0.6]

            def gmf(dikm, fi=fi):
                return raw_gmkm(3.0, 3.5, dikm, flat, fi)

            r = gmf(15000.0) / 15000.0
            if want_left_over:
                guess, expr = 1000.0 / (1.0 - r), (
                    lambda x: f32(f32(x - gmf(x)) - f32(1000.0)))
            else:
                guess, expr = 1000.0 / r, (
                    lambda x: f32(gmf(x) - f32(1000.0)))
            hit = search(guess, expr, -3000, 3000)
            if hit is not None:
                return fi, hit
        return None, None

    for want_left_over in (False, True):
        fi, hit = find_dikm_on(want_left_over)
        assert hit is not None, f"no exact 1000 km hit ({want_left_over})"
        for n in (-40000, -4000, -1, 0, 1, 4000, 40000):
            rows.append(case(fi1=fi[0], fi2=fi[1], fi3=fi[2],
                             dikm=step32(hit, n)))
    # --- DIKM: negative, zero, and a wide sweep ---------------------------
    for dikm in (-1.0, -0.5, 0.0, 1.0, 500.0, 1000.0, 2000.0, 4000.0,
                 8000.0, 12000.0, 20000.0, 40000.0):
        rows.append(case(dikm=dikm))

    # --- the sin^2 clamp at 0.9999 (unreachable with physical geometry) ---
    for delopt, ht in ((0.0, 0.0), (0.0, 0.1), (0.0, 0.3), (0.0, 0.32),
                       (0.0, 0.4), (0.0, 1.0), (3.0, -20.0), (3.0, -9.0),
                       (3.0, -8.0), (3.0, -5.0), (3.0, 0.0), (3.0, 110.0)):
        rows.append(case(delopt=delopt, hi1=ht, hi2=ht, hi3=ht,
                         freq=0.02, fi1=0.004, fi2=0.006, fi3=0.004))
        rows.append(case(delopt=delopt, hi1=ht, freq=0.02,
                         fi1=0.004, fi2=0.006, fi3=0.004))

    # --- heights differing between areas ----------------------------------
    for hi in ([100.0, 110.0, 120.0], [120.0, 110.0, 100.0],
               [90.0, 130.0, 90.0], [130.0, 90.0, 130.0],
               [110.0, 110.0, 300.0]):
        rows.append(case(hi1=hi[0], hi2=hi[1], hi3=hi[2]))

    # --- FI zero, and DELOPT across a wide range --------------------------
    rows.append(case(fi1=0.0, fi2=0.0, fi3=0.0, freq=0.0))
    rows.append(case(fi1=0.0, fi2=0.8, fi3=0.0, freq=1.0))
    rows.append(case(fi1=0.6, fi2=0.0, fi3=0.6))
    for delopt in (0.0, 1.0, 3.0, 10.0, 30.0, 60.0, 89.0):
        rows.append(case(delopt=delopt))
        rows.append(case(delopt=delopt, freq=1.2))

    # --- random fill for breadth ------------------------------------------
    state = SEED

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    for _ in range(300):
        rows.append(case(
            delopt=rnd(0.0, 20.0),
            freq=rnd(0.5, 12.0),
            dikm=rnd(-1.0, 25000.0),
            hi1=rnd(85.0, 140.0), hi2=rnd(85.0, 140.0), hi3=rnd(85.0, 140.0),
            fi1=rnd(0.0, 2.0), fi2=rnd(0.0, 2.0), fi3=rnd(0.0, 2.0),
        ))
    # a tighter band around the admissible window, so the pass arm is common
    for _ in range(200):
        f2 = rnd(0.7, 1.1)
        rows.append(case(
            freq=rnd(3.0, 6.0),
            dikm=rnd(2000.0, 22000.0),
            fi1=rnd(0.4, 0.7), fi2=f2, fi3=rnd(0.4, 0.7),
        ))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "gettop_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"gettop: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
