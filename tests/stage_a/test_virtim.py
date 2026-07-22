"""Stage A REDMAP+VIRTIM vs. the tier-B oracle dump.

The chain under test: ASCII coefficient ingestion (Module 0) -> REAL*4
sunspot blend (REDMAP) -> diurnal Fourier collapse (VIRTIM).  Measured at
generation: all 96 cases x 318 AB coefficients bit-exact (30,528 values)
— the sin/cos of the single time angle come out identical and everything
else is plain float32 arithmetic.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_a.virtim import virtim

from .dumputil import COEFFS_DIR, load_dump, ulp_diff

DUMP = load_dump("virtim")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_ionomap_dumps.sh")
def test_virtim_vs_oracle():
    header, inputs, rows = DUMP
    total = 0
    exact = 0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        coeff = "URSI" if int(inp[0]) == 1 else "CCIR"
        st = redmap(COEFFS_DIR, coeff, int(inp[1]), F32(inp[2]))
        ab = virtim(st, F32(inp[3]))
        assert len(exp) == 318
        for i in range(318):
            d = ulp_diff(ab[i + 1], np.float32(exp[i]))
            assert d <= 1.0, (
                f"case {k} ab({i + 1}): got {ab[i + 1]!r} expected "
                f"{np.float32(exp[i])!r} ({d:.2f} ULP), in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.999, f"bit-exact rate {exact}/{total}"
