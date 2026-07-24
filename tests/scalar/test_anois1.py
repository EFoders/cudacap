"""Stage A ANOIS1 vs the tier-B oracle dump (1 MHz atmospheric noise).

Each case loads the NOISY maps via redmap(), injects the receiver mean time and
location, and runs ANOIS1.  The block indices/times CC/TM/KJ/JK are integer/
exact; the two map values ATNU/ATNY are NOISY Fourier sums (blocks 1..6, the
full 29x15 series) and carry the [[f32-libm-noise]] SIN/COS floor accumulated
over the series.  For the no-data-base (IZF) cases only ATNU/ATNY (both 0) are
checked -- the Fortran leaves CC/TM/KJ/JK at stale values on that early return.

Measured over 189 cases = 1098 checked cells: 98.5% bit-exact, worst 2.0
floored ULP (an ATNU ~63.6 differing by 8e-6, the accumulated series floor).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from voacap_gpu.scalar.anois1 import anois1
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.noisestate import AnoisState
from voacap_gpu.scalar.redmap import redmap

from .dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("anois1")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_anois1_dumps.sh")
def test_anois1_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["atnu", "atny", "cc", "tm", "kj", "jk"]
    total = exact = 0
    worst = 0.0
    worst_where = ""
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        month = int(inp[0])
        ssn, gmtr, rlat, rlong, rlongd = (float(x) for x in inp[1:6])
        izf = int(inp[6])
        st = redmap(COEFFS_DIR, "CCIR", month, F32(ssn))
        if izf:  # force the no-data branch without touching the cache
            st = SimpleNamespace(f2d=np.zeros((1, 1, 1), dtype=np.float32))
        a = AnoisState()
        anois1(a, st, gmtr, rlong, rlongd, rlat)
        got = [a.atnu, a.atny, a.cc, a.tm, a.kj, a.jk]
        # ATNU/ATNY always checked; CC/TM/KJ/JK only when data is present
        checks = [True, True, not izf, not izf, not izf, not izf]
        for name, g, e_str, chk in zip(header, got, exp, checks):
            if not chk:
                continue
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            if d > worst:
                worst, worst_where = d, f"case {k} {name}: {g!r} vs {e!r}"
            assert d <= 4, (f"case {k} {name}: got {g!r} expected {e!r} "
                            f"({d:.2f} floored ULP) in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.95, (f"exact {exact}/{total} "
                                   f"(worst {worst:.2f} @ {worst_where})")
