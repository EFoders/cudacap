"""Stage A PRBMUF vs the tier-B oracle dump (MUF-exceedance probability).

PRBMUF is pure float32 arithmetic -- a quartic Horner polynomial, a reciprocal
and a couple of divisions, with no transcendentals -- so every operation is
correctly rounded in numpy and gfortran alike and the result is bit-exact.  The
247 cases span the FSET<=0 step branch, the FSET>0 normal-approximation branch
on both sides of the median (SIGL vs SIGU), the |Z|<=5 clamp and the SIG=0.001
floor.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.mufstate import MufState
from voacap_gpu.stage_a.prbmuf import prbmuf

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("prbmuf")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_prbmuf_dumps.sh")
def test_prbmuf_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["prbmuf"]
    total = exact = 0
    worst = 0.0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        fmhz, fgo, fset, il_s, siglil, siguil = inp
        il = int(il_s)
        m = MufState()
        m.sigl[il] = F32(float(siglil))
        m.sigu[il] = F32(float(siguil))
        got = prbmuf(m, float(fmhz), float(fgo), float(fset), il)
        e = np.float32(exp[0])
        d = ulp_diff_floored(got, e)
        worst = max(worst, d)
        assert d <= 1, (f"case {k}: got {got!r} expected {e!r} "
                        f"({d:.2f} floored ULP) in {inp}")
        total += 1
        exact += d == 0.0
    assert exact / total >= 0.999, f"exact {exact}/{total} (worst {worst:.2f})"
