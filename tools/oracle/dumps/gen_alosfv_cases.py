#!/usr/bin/env python3
"""Deterministic case grid for the ALOSFV (Module-4 deviative-loss) dump
(stdlib-only).

ALOSFV depends on the ionogram (FVERT/HTRUE/HPRIM built from injected
FI/YI/HI + IEDP) and the per-layer MUF heights HPMUF/HTMUF (injected directly,
isolating ALOSFV from CURMUF).  HTMUF is sampled so both branches of each
height comparison fire: HTMUF(1) below the E-layer true heights (keeps CF
defined), HTMUF(2)/HTMUF(3) in the F1/F2 range so some HTRUE points sit above
and some below.  HPMUF = HTMUF + a positive virtual-height excess.

Usage: gen_alosfv_cases.py <outdir>
alosfv_cases.csv:
  fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3,iedp,
  hpmuf1,hpmuf2,hpmuf3,htmuf1,htmuf2,htmuf3
"""

import pathlib
import sys

SEED = 20260722


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def cases():
    rows = []
    rnd = lcg(SEED)
    for i in range(160):
        fi1 = rnd(1.0, 4.5)
        f1_present = rnd(0.0, 1.0) < 0.55
        fi3 = rnd(max(fi1 + 1.0, 4.0), 15.0)
        if f1_present:
            fi2 = rnd(fi1 + 0.3, min(fi1 + 3.5, fi3 - 0.3))
            yi2 = rnd(30.0, 52.0)
            hi2 = rnd(160.0, 225.0)
        else:
            fi2 = yi2 = hi2 = 0.0
        yi1 = rnd(17.0, 23.0)
        yi3 = rnd(55.0, 145.0)
        hi1 = rnd(105.0, 115.0)
        hi3 = rnd(max(280.0, hi2 + 60.0), 490.0)
        iedp = -1 if rnd(0.0, 1.0) < 0.5 else 0
        # MUF heights: HTMUF(1) below E true heights; F1/F2 spread over the
        # reflection range so the HTRUE-vs-HTMUF branches both fire.
        htmuf1 = rnd(75.0, 100.0)
        htmuf2 = rnd(150.0, 240.0)
        htmuf3 = rnd(250.0, min(430.0, hi3 - 10.0))
        hpmuf1 = htmuf1 + rnd(8.0, 40.0)
        hpmuf2 = htmuf2 + rnd(10.0, 70.0)
        hpmuf3 = htmuf3 + rnd(10.0, 90.0)
        rows.append(",".join(f(v) for v in
                    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, iedp,
                     hpmuf1, hpmuf2, hpmuf3, htmuf1, htmuf2, htmuf3)))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "alosfv_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"alosfv: {len(rows)} cases")


if __name__ == "__main__":
    main()
