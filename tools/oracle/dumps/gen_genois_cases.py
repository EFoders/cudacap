#!/usr/bin/env python3
"""Deterministic case grid for the GENOIS (combined noise) dump.

Everything GENOIS reads apart from the /TWO/ tables is a plain scalar, so
every case is injected.  The branches that have to be reached:

* **NOISE > 0** -- an explicit 3 MHz level in dB below 1 W (the card's usual
  form, 130..170); **NOISE = 0** -- the -164 dBW default, which routes to
  category 4; **NOISE < 0** -- a CCIR environment index, including -5 and
  below, which MIN0(4,MA) clamps back to 4.
* **IZF = 1** -- F2D(1,1,1) forced negative, i.e. no long-term data base:
  GENFAM is skipped entirely and fixed deciles are substituted.
* **FREQ <= FI(3,KFX)** -- galactic noise cannot penetrate the F2 layer, so
  the whole galactic component is zeroed and MODE_N2 goes to 0.  Cases sit
  either side of and exactly on FI3.
* **Caruana's limit** -- reached only when GENFAM returns DUA or DLA above
  12 dB, which the real DUD tables do only at low frequency in the
  southern-hemisphere time blocks (RLAT < 0, blocks 2..5, below ~7 MHz).
  Those cases are placed deliberately; the test asserts they were hit.
* **SLOP** -- |CC - TM|/4, the weight between the two local-time blocks,
  swept over 0 (KJ == JK, no interpolation), a half block, and a full one.
* **GAIN match / no match** -- FREQ inside and outside [RXLO,RXHI], the
  latter leaving all seven EFF slots at 0.

FREQ is kept at or above 1 MHz throughout: GAIN indexes ARRAY at INT(FMC)
with no bottom clamp, and RXHI is kept at or below 30 for the same reason at
the top, so no case depends on the Fortran's out-of-bounds read.

stdlib-only.  Usage: gen_genois_cases.py <outdir>
genois_cases.csv columns:
    month,izf,freq,noise,rlat,atnu,atny,cc,tm,kj,jk,fi3,rxlo,rxhi
"""

import pathlib
import sys

SEED = 20260723

RX = (2.0, 30.0)          # receiver antenna range that matches 2..30 MHz
NORX = (12.0, 18.0)       # a narrow range, so most frequencies miss


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def row(month, izf, freq, noise, rlat, atnu, atny, cc, tm, kj, jk, fi3, rx):
    return ",".join([
        str(int(month)), str(int(izf)), f(freq), str(int(noise)), f(rlat),
        f(atnu), f(atny), f(cc), f(tm), str(int(kj)), str(int(jk)),
        f(fi3), f(rx[0]), f(rx[1])])


def cases():
    rows = []
    # --- man-made noise: every arm of the IF (MAN) 120,114,115 ------------
    for noise in (170, 155, 145, 130, 0, -1, -2, -3, -4, -5, -9):
        for freq in (3.0, 7.0, 15.0, 28.0):
            rows.append(row(6, 0, freq, noise, 0.6, 70.0, 65.0,
                            10.0, 10.0, 3, 3, 5.0, RX))
    # --- no long-term data base -------------------------------------------
    for month in (1, 5, 11):
        for noise in (145, 0, -2):
            rows.append(row(month, 1, 9.0, noise, 0.4, 70.0, 65.0,
                            13.0, 14.0, 4, 4, 6.0, RX))
            rows.append(row(month, 1, 4.0, noise, -0.4, 90.0, 85.0,
                            2.0, 2.0, 1, 1, 6.0, RX))
    # --- galactic cut-off: FREQ either side of and exactly on FI(3,KFX) ---
    for fi3 in (4.0, 9.0, 14.0):
        for freq in (fi3 - 1.0, fi3 - 0.01, fi3, fi3 + 0.01, fi3 + 1.0):
            rows.append(row(7, 0, freq, -2, 0.5, 75.0, 70.0,
                            8.0, 10.0, 3, 2, fi3, RX))
    # --- Caruana's limit: southern blocks at low frequency ----------------
    for kj in (2, 3, 4, 5):
        for freq in (1.0, 2.0, 3.0, 5.0, 7.0):
            for month in (1, 4, 7, 10):
                rows.append(row(month, 0, freq, -1, -0.9, 95.0, 92.0,
                                4.0 * kj - 2.0, 4.0 * kj - 2.0, kj, kj,
                                2.5, RX))
    # --- SLOP: no interpolation, half a block, a whole one -----------------
    for kj, jk in ((3, 3), (3, 2), (3, 4), (6, 1), (1, 6)):
        tm = 4.0 * kj - 2.0
        for cc in (tm, tm + 1.0, tm - 1.0, tm + 2.0, tm - 4.0):
            rows.append(row(9, 0, 11.0, -3, 0.2, 80.0, 40.0,
                            cc, tm, kj, jk, 7.0, RX))
    # --- antenna match vs no match, and the 55 MHz DUME cap ---------------
    for freq in (1.5, 2.0, 11.9, 12.0, 18.0, 18.1, 30.0, 40.0, 55.0):
        rows.append(row(3, 0, freq, -2, 0.5, 70.0, 68.0,
                        6.0, 6.0, 2, 2, 5.0, NORX))
        rows.append(row(3, 0, min(freq, 30.0), -2, 0.5, 70.0, 68.0,
                        6.0, 6.0, 2, 2, 5.0, RX))
    # --- ATNU/ATNY range, both hemispheres --------------------------------
    for atnu in (0.0, 20.0, 55.0, 90.0, 130.0):
        for rlat in (1.1, -1.1):
            rows.append(row(12, 0, 6.5, -1, rlat, atnu, atnu - 5.0,
                            15.0, 14.0, 4, 3, 4.0, RX))
    # --- random fill for breadth ------------------------------------------
    rnd = lcg(SEED)
    for _ in range(360):
        month = 1 + int(rnd(0.0, 12.0)) % 12
        izf = 1 if rnd(0.0, 1.0) < 0.05 else 0
        freq = rnd(1.0, 40.0)
        r = rnd(0.0, 1.0)
        noise = int(rnd(120.0, 175.0)) if r < 0.4 else -(1 + int(rnd(0.0, 6.0)))
        if 0.38 < r < 0.4:
            noise = 0
        rlat = rnd(-1.5, 1.5)
        atnu = rnd(0.0, 140.0)
        atny = atnu + rnd(-25.0, 25.0)
        kj = 1 + int(rnd(0.0, 6.0)) % 6
        tm = 4.0 * kj - 2.0
        cc = tm + rnd(-4.0, 4.0)
        jk = 1 + int(rnd(0.0, 6.0)) % 6
        fi3 = rnd(2.0, 16.0)
        rx = RX if rnd(0.0, 1.0) < 0.75 else NORX
        rows.append(row(month, izf, freq, noise, rlat, atnu, atny,
                        cc, tm, kj, jk, fi3, rx))
    # group by month: the driver reloads /TWO/ only when the month changes
    rows.sort(key=lambda r: int(r.split(",", 1)[0]))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "genois_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"genois: {len(rows)} cases")


if __name__ == "__main__":
    main()
