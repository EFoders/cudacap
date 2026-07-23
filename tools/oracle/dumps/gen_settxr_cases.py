#!/usr/bin/env python3
"""Deterministic case grid for the SETTXR dump.

SETTXR prices all 45 rays at each of the two long-path control areas.  The ray
tables are far too wide to inject value by value, so each area's six /REFLX/
columns are generated as a LINEAR RAMP -- a base and a step -- and the number
of filled rays is injected separately.  That is not a loss of generality for
the branch structure: every test in the routine is a threshold on one ray
column, so a ramp that crosses the threshold puts rays on both sides of it
inside a single case, and sweeping the base moves the crossing point.

What has to be reached:

* **the ray-existence scan's three exits.**  It stops at the FIRST ray whose
  HPFLX is under 70 km and takes the over-the-MUF fallback; it skips a ray
  whose hop count is under 0.9 and keeps looking; and it stops with a mode
  when one clears the minimum take-off angle.  Falling off the end of all 45
  also lands on the fallback.
* **the over-the-MUF fallback firing and not firing**, since it OVERWRITES
  ray 1 of /REFLX/ -- the dump reads ray 1 back to see whether it did.
* **HPFLX and DELFX either side of their tests in the pricing loop**, which
  are the same two thresholds but written as .lt. rather than as arithmetic
  IFs, so a ray can be scanned as existing and then skipped as not.
* **FI(1,K) either side of FVFLX**, the D-E versus F layer split.
* **HTLOSS either side of HTFLX**, the two collision-frequency arms, and
  HTFLX either side of the 100 km HEFF clamp.
* **FVFLX/FI(1,K) either side of XVE**, the AMAX1 in the E-layer correction.
* **FS(2,K) at and below zero**, the sporadic-E obscuration and its absence,
  plus P clamped at both 0.1 and 0.9.
* **KM at 1, 3 and 5**, since the ground loss averages over that many areas.
* **GAIN finding an antenna entry and not finding one** (frequency outside
  the injected range), which returns a gain of 0.
* **MODMUF 1..4**.

Held away from the grid, and why:

* MODMUF outside 1..4 indexes DELMUF and YMUF off the end of four-element
  arrays.  The port raises; the oracle is never asked to read out of bounds.
* GDFLX at zero divides by zero in the hop count.  The ramps are kept inside
  [100, 5000] km.
* HTFLX at or below zero puts SINP at or above 1 and SQRT(1-SINP^2) into the
  negatives -- and it is also the only way to reach CPHET's 1e-8 floor, which
  is therefore unreachable for any live ray.
* FI(1,K) at zero divides by zero in XV; HS(K) at zero does the same in SECS
  for a ray launched horizontally.  Both kept positive.
* HNU at zero divides by zero in the collision frequency, XVE at or below
  zero puts ALOG into the negatives, and YMUF at or below zero sends PRBMUF
  into its step branch where -10*ALOG10(P) can be infinite.  All kept
  positive.

stdlib-only.  Usage: gen_settxr_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260730

INTS = ["itxrcp2", "km", "modmuf", "nhopmf", "nfill1", "nfill2",
        "ifob13_1", "ifob13_2"]
RAMP = ["delfx0", "ddelfx", "hpflx0", "dhpflx", "htflx0", "dhtflx",
        "gdflx0", "dgdflx", "fvflx0", "dfvflx", "afflx0", "dafflx"]
FLOATS = (
    ["freq", "gcdkm", "amind", "htloss", "xnuz", "hnu", "afe", "bfe", "xve"]
    + ["fi1_1", "fi1_2", "gyz_1", "gyz_2", "abiy_1", "abiy_2",
       "fs2_1", "fs2_2", "hs_1", "hs_2"]
    + [f"sigpat{j}" for j in range(1, 6)]
    + [f"epspat{j}" for j in range(1, 6)]
    + [f"{a}{j}" for a in ("ymuf", "sigl", "sigu") for j in (1, 2, 3, 4)]
    + ["delmuf", "hpmuf", "htmuf", "fvmufm", "afmuf"]
    + [f"{r}_1" for r in RAMP] + [f"{r}_2" for r in RAMP]
    + ["txlo", "txhi", "rxlo", "rxhi", "pwrdba"]
)
COLUMNS = INTS + FLOATS

DEFAULTS = {
    "itxrcp2": 3, "km": 5, "modmuf": 3, "nhopmf": 4, "nfill1": 12,
    "nfill2": 12, "ifob13_1": 4200, "ifob13_2": 3800,
    "freq": 14.0, "gcdkm": 15000.0, "amind": 3.0, "htloss": 90.0,
    "xnuz": 12.0, "hnu": 5.0, "afe": 0.7, "bfe": 1.3, "xve": 0.5,
    "fi1_1": 3.0, "fi1_2": 2.6, "gyz_1": 1.358, "gyz_2": 1.1,
    "abiy_1": 0.35, "abiy_2": 0.3, "fs2_1": 4.0, "fs2_2": 3.0,
    "hs_1": 110.0, "hs_2": 110.0,
    "delmuf": 6.0, "hpmuf": 320.0, "htmuf": 280.0, "fvmufm": 6.5,
    "afmuf": 1.1,
    "txlo": 2.0, "txhi": 30.0, "rxlo": 2.0, "rxhi": 30.0, "pwrdba": 40.0,
}
for _j, _s, _e in ((1, 0.005, 15.0), (2, 0.02, 12.0), (3, 0.001, 20.0),
                   (4, 0.01, 10.0), (5, 0.003, 25.0)):
    DEFAULTS[f"sigpat{_j}"] = _s
    DEFAULTS[f"epspat{_j}"] = _e
for _j, _v in enumerate([26.0, 24.0, 22.0, 12.0], 1):
    DEFAULTS[f"ymuf{_j}"] = _v
    DEFAULTS[f"sigl{_j}"] = 1.8
    DEFAULTS[f"sigu{_j}"] = 1.4
# the default ramps: rays climbing away from the horizon, virtual height and
# vertical frequency rising with them, ground distance falling.  The steps are
# chosen so that even a full 45-ray table stays inside every legal range --
# take-off angle under 90 degrees (GAIN's elevation axis stops there), true
# height positive (SQRT(1-SINP^2)), ground distance positive (the hop count
# divides by it), vertical frequency above -GYZ (powf of a negative base).
_BASE_RAMP = {"delfx0": 2.0, "ddelfx": 1.8, "hpflx0": 60.0, "dhpflx": 8.0,
              "htflx0": 70.0, "dhtflx": 5.0, "gdflx0": 4000.0,
              "dgdflx": -80.0, "fvflx0": 1.5, "dfvflx": 0.9,
              "afflx0": 0.8, "dafflx": 0.15}
for _k, _v in _BASE_RAMP.items():
    DEFAULTS[f"{_k}_1"] = _v
    DEFAULTS[f"{_k}_2"] = _v


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def case(**kw) -> str:
    unknown = set(kw) - set(COLUMNS)
    if unknown:
        raise KeyError(f"not a column: {sorted(unknown)}")
    d = dict(DEFAULTS)
    d.update(kw)
    return ",".join(str(int(d[c])) if c in INTS else f(d[c])
                    for c in COLUMNS)


def both(**kw):
    """Apply the same ramp override to both control areas."""
    out = {}
    for key, val in kw.items():
        out[f"{key}_1"] = val
        out[f"{key}_2"] = val
    return out


def _safe(v0, dv, nfill, lo, hi, fallback):
    """Keep a ramp inside [lo, hi] over NFILL rays, or fall back.

    The bounds are not cosmetic.  A take-off angle past 90 degrees runs off
    GAIN's elevation axis, a true height at or below zero puts SQRT(1-SINP^2)
    into the negatives, a ground distance at zero divides by zero in the hop
    count, and a vertical frequency below -GYZ makes powf return NaN.  All
    four are out-of-bounds or NaN in the Fortran too, so the oracle is never
    asked to evaluate them.
    """
    end = v0 + dv * (max(nfill, 1) - 1)
    if lo <= v0 <= hi and lo <= end <= hi:
        return v0, dv
    return fallback


def cases():
    rows = []
    # --- the ray-existence scan: where it stops --------------------------
    # ray 1 already dead -> the fallback fires immediately
    for h0 in (0.0, 10.0, 69.9, 70.0, 70.1, 100.0):
        rows.append(case(**both(hpflx0=h0, dhpflx=0.0)))
        rows.append(case(**both(hpflx0=h0, dhpflx=22.0)))
    # no ray clears the minimum angle -> falls off the end -> fallback
    for amind in (0.0, 3.0, 6.0, 20.0, 60.0, 200.0):
        rows.append(case(amind=amind))
        rows.append(case(amind=amind, **both(delfx0=0.0, ddelfx=0.0)))
    # every ray under 0.9 hops -> the scan skips them all -> fallback
    for gd in (200.0, 4000.0, 4444.0, 4445.0, 5000.0):
        rows.append(case(**both(gdflx0=gd, dgdflx=0.0)))
        rows.append(case(gcdkm=3000.0, **both(gdflx0=gd, dgdflx=0.0)))
    # --- NFILL: none, one, some, all -------------------------------------
    for n in (0, 1, 2, 5, 12, 20, 44, 45):
        rows.append(case(nfill1=n, nfill2=n))
        rows.append(case(nfill1=n, nfill2=12))
    # --- the D-E versus F layer split ------------------------------------
    for fi1 in (0.5, 1.4, 1.5, 1.6, 3.0, 6.0, 12.0, 40.0):
        rows.append(case(fi1_1=fi1, fi1_2=fi1))
        rows.append(case(fi1_1=fi1, fi1_2=fi1,
                         **both(fvflx0=1.5, dfvflx=0.0)))
    # --- HTLOSS either side of HTFLX, and the 100 km HEFF clamp ----------
    for htloss in (0.0, 60.0, 70.0, 84.0, 90.0, 98.0, 200.0, 400.0):
        rows.append(case(htloss=htloss))
        rows.append(case(htloss=htloss, **both(htflx0=95.0, dhtflx=2.0)))
        rows.append(case(htloss=htloss, **both(htflx0=200.0, dhtflx=0.0)))
    # --- XVE either side of FVFLX/FI(1,K) --------------------------------
    for xve in (0.001, 0.1, 0.5, 1.0, 2.0, 8.0):
        rows.append(case(xve=xve, fi1_1=12.0, fi1_2=12.0))
    # --- sporadic E present and absent, and P at both clamps -------------
    for fs2 in (-1.0, 0.0, 0.5, 2.0, 4.0, 9.0, 30.0):
        rows.append(case(fs2_1=fs2, fs2_2=fs2, fi1_1=0.5, fi1_2=0.5))
        rows.append(case(fs2_1=fs2, fs2_2=fs2, fi1_1=0.5, fi1_2=0.5,
                         freq=3.0))
    for ifob in (0, 1000, 4200, 26000, 60000):
        rows.append(case(ifob13_1=ifob, ifob13_2=ifob, fi1_1=0.5,
                         fi1_2=0.5))
    # --- KM: the ground-loss average over 1, 3 and 5 areas ---------------
    for km in (1, 3, 5):
        rows.append(case(km=km))
        rows.append(case(km=km, fi1_1=0.5, fi1_2=0.5))
    # --- MODMUF, and the fallback's own numbers --------------------------
    for m in (1, 2, 3, 4):
        rows.append(case(modmuf=m, **both(hpflx0=0.0, dhpflx=0.0)))
        rows.append(case(modmuf=m))
    for nh in (1, 2, 4, 9, 20):
        rows.append(case(nhopmf=nh, **both(hpflx0=0.0, dhpflx=0.0)))
    # --- GAIN inside and outside the antenna's frequency range -----------
    for freq, lo, hi in ((14.0, 2.0, 30.0), (1.0, 2.0, 30.0),
                         (31.0, 2.0, 30.0), (14.0, 20.0, 30.0),
                         (14.0, 2.0, 10.0)):
        rows.append(case(freq=freq, txlo=lo, txhi=hi))
        rows.append(case(freq=freq, rxlo=lo, rxhi=hi))
    # --- the receiver control area at 2 and at 3 -------------------------
    for itx2 in (2, 3):
        rows.append(case(itxrcp2=itx2))
        rows.append(case(itxrcp2=itx2, fi1_1=0.5, fi1_2=0.5))
    # --- random fill for breadth -----------------------------------------
    rnd = lcg(SEED)
    for _ in range(150):
        n1 = int(rnd(0.0, 46.0)) % 46
        n2 = int(rnd(0.0, 46.0)) % 46
        ramp = {}
        for suff, nf in (("_1", n1), ("_2", n2)):
            d0, dd = _safe(rnd(0.0, 20.0), rnd(-0.5, 3.0), nf,
                           -40.0, 89.0, (2.0, 1.8))
            h0, dh = _safe(rnd(0.0, 200.0), rnd(-4.0, 24.0), nf,
                           -300.0, 900.0, (60.0, 8.0))
            t0, dt = _safe(rnd(60.0, 300.0), rnd(-2.0, 12.0), nf,
                           40.0, 600.0, (70.0, 5.0))
            g0, dg = _safe(rnd(150.0, 4800.0), rnd(-120.0, 40.0), nf,
                           100.0, 5000.0, (4000.0, -80.0))
            f0, df = _safe(rnd(0.2, 12.0), rnd(-0.1, 1.2), nf,
                           0.05, 45.0, (1.5, 0.9))
            a0, da = _safe(rnd(0.0, 3.0), rnd(-0.02, 0.3), nf,
                           -1.0, 20.0, (0.8, 0.15))
            ramp.update({
                f"delfx0{suff}": d0, f"ddelfx{suff}": dd,
                f"hpflx0{suff}": h0, f"dhpflx{suff}": dh,
                f"htflx0{suff}": t0, f"dhtflx{suff}": dt,
                f"gdflx0{suff}": g0, f"dgdflx{suff}": dg,
                f"fvflx0{suff}": f0, f"dfvflx{suff}": df,
                f"afflx0{suff}": a0, f"dafflx{suff}": da,
            })
        kw = {
            "itxrcp2": 2 + int(rnd(0.0, 2.0)) % 2,
            "km": (1, 3, 5)[int(rnd(0.0, 3.0)) % 3],
            "modmuf": 1 + int(rnd(0.0, 4.0)) % 4,
            "nhopmf": 1 + int(rnd(0.0, 12.0)) % 12,
            "nfill1": n1, "nfill2": n2,
            "ifob13_1": int(rnd(0.0, 40000.0)),
            "ifob13_2": int(rnd(0.0, 40000.0)),
            "freq": rnd(1.5, 34.0), "gcdkm": rnd(600.0, 20000.0),
            "amind": rnd(0.0, 30.0), "htloss": rnd(50.0, 250.0),
            "xnuz": rnd(1.0, 40.0), "hnu": rnd(1.0, 20.0),
            "afe": rnd(-2.0, 3.0), "bfe": rnd(-3.0, 3.0),
            "xve": rnd(0.01, 3.0),
            "fi1_1": rnd(0.3, 14.0), "fi1_2": rnd(0.3, 14.0),
            "gyz_1": rnd(0.3, 2.0), "gyz_2": rnd(0.3, 2.0),
            "abiy_1": rnd(0.0, 1.5), "abiy_2": rnd(0.0, 1.5),
            "fs2_1": rnd(-1.0, 12.0), "fs2_2": rnd(-1.0, 12.0),
            "hs_1": rnd(90.0, 130.0), "hs_2": rnd(90.0, 130.0),
            "delmuf": rnd(0.0, 40.0), "hpmuf": rnd(90.0, 450.0),
            "htmuf": rnd(70.0, 380.0), "fvmufm": rnd(1.0, 18.0),
            "afmuf": rnd(0.2, 3.0),
            # the antenna tables stop at 30 MHz; a range that admitted a
            # higher frequency would index ARRAY past its 30th row
            "txlo": rnd(1.0, 12.0), "txhi": rnd(13.0, 30.0),
            "rxlo": rnd(1.0, 12.0), "rxhi": rnd(13.0, 30.0),
            "pwrdba": rnd(20.0, 55.0),
            **ramp,
        }
        for j in (1, 2, 3, 4):
            kw[f"ymuf{j}"] = rnd(4.0, 42.0)
            kw[f"sigl{j}"] = rnd(0.2, 6.0)
            kw[f"sigu{j}"] = rnd(0.2, 6.0)
        for j in range(1, 6):
            kw[f"sigpat{j}"] = rnd(-0.001, 0.05)
            kw[f"epspat{j}"] = rnd(1.0, 30.0)
        rows.append(case(**kw))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "settxr_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"settxr: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
