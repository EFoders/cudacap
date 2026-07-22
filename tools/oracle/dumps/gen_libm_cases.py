#!/usr/bin/env python3
"""Argument grid for the raw-libm dump (stdlib-only).

Emits libm_cases.csv: one REAL*4 value per line, chosen to probe the
sinf/cosf/acosf/atanf argument reductions across sign, magnitude and the
range-reduction quadrant boundaries near multiples of pi/4.

Values are written as exact float32 decimal via struct round-trip so the
Fortran READ and the Python emulator consume identical bits.
"""

import math
import pathlib
import struct
import sys


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def line(x):
    return repr(f32(x))


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def main():
    out = pathlib.Path(sys.argv[1])
    vals = []
    # near the pi/4 * k reduction boundaries (where cosf/sinf branch)
    for k in range(-16, 17):
        b = k * math.pi / 4
        for d in (-2e-6, -1e-7, 0.0, 1e-7, 2e-6):
            vals.append(b + d)
    # small args (polynomial path, no reduction)
    for e in range(-20, 2):
        vals.append(2.0**e)
        vals.append(-(2.0**e))
    # acos domain [-1,1] densely; atan across scales
    rnd = lcg(20260724)
    for _ in range(300):
        vals.append(rnd(-1.0, 1.0))
    for _ in range(300):
        vals.append(rnd(-math.pi, math.pi))
    for _ in range(200):
        vals.append(rnd(-20.0, 20.0))
    # exact endpoints
    vals += [-1.0, 1.0, 0.0, -0.0, math.pi / 2, -math.pi / 2]
    seen = set()
    rows = []
    for v in vals:
        b = struct.pack("f", v)
        if b in seen:
            continue
        seen.add(b)
        rows.append(line(v))
    out.write_text("\n".join(rows) + "\n", newline="\n")
    print(f"libm: {len(rows)} cases")


if __name__ == "__main__":
    main()
