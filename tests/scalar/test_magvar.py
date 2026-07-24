"""Stage A MAGVAR/MAGFIN vs. the tier-B oracle dump.

Measured on generation (gfortran 15/glibc vs Windows CPython): gyz and clg
100% bit-exact, gmdip 233/234 with worst 1 ULP — the f32 sin/cos/atan
chains agree to correct rounding essentially everywhere.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.magfin import magvar

from .dumputil import load_dump, ulp_diff

DUMP = load_dump("magvar")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_mod1_dumps.sh")
def test_magvar_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["gyz", "gmdip", "clg"]
    total = 0
    exact = 0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = magvar(F32(inp[0]), F32(inp[1]))
        for h, gv, ev in zip(header, got, exp):
            d = ulp_diff(gv, np.float32(ev))
            assert d <= 2.0, (f"case {k} {h}: got {gv!r} expected "
                              f"{np.float32(ev)!r} ({d:.2f} ULP), in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.99, f"bit-exact rate {exact}/{total}"
