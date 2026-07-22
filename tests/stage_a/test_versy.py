"""Stage A VERSY vs. the tier-B oracle dump (full REDMAP->VIRTIM->VERSY
chain per case).

Measured at generation: 1336/1344 gamma values bit-exact.  The eight
misses trace to ONE mechanism, fully diagnosed: GOB = COS(CENLAT) is a
runtime glibc cosf call in the oracle, and glibc's binary32 cosf is
within 1 ULP but NOT correctly rounded — for cenlat=-0.825022324 it
returns 1 ULP above the correctly-rounded value our f32math produces
(0x3F2DB4D6 vs 0x3F2DB4D5).  GOB scales every cos^m(lat) block of the
geographic series, and the Es sets amplify the seed through massive
cancellation (partial sums swing to ~|660| for a ~1.7 result, a ~350x
factor), giving the observed worst |diff| ~5e-5 on gamma1..3.  gamma5/6
(smaller cancellation) stay at <=2 ULP.  The planned fix is an exact
emulation of glibc's sinf/cosf (see docs/f32_libm.md); until then the
bars below encode the measured amplification, and the rate guardrail
keeps them from hiding structural regressions.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_a.versy import versy
from voacap_gpu.stage_a.virtim import virtim

from .dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("versy")

# scale-floored ULP bars per gamma index (measured worst: 397 on gamma1)
BARS = {1: 512, 2: 512, 3: 512, 4: 128, 5: 8, 6: 8}


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_ionomap_dumps.sh")
def test_versy_vs_oracle():
    header, inputs, rows = DUMP
    assert header == [f"gamma{i}" for i in range(1, 7)]
    total = 0
    exact = 0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        coeff = "URSI" if int(inp[0]) == 1 else "CCIR"
        st = redmap(COEFFS_DIR, coeff, int(inp[1]), F32(inp[2]))
        ab = virtim(st, F32(inp[3]))
        gamma = versy(st, ab, F32(inp[4]), F32(inp[5]), F32(inp[6]), 1, 6)
        for i in range(1, 7):
            e = np.float32(exp[i - 1])
            d = ulp_diff_floored(gamma[i], e)
            assert d <= BARS[i], (
                f"case {k} gamma{i}: got {gamma[i]!r} expected {e!r} "
                f"({d:.2f} floored ULP > {BARS[i]}), in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.99, f"bit-exact rate {exact}/{total}"
