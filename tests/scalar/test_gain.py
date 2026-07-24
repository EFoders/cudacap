"""Stage A GAIN ground-reflection branch vs the tier-B oracle dump.

GAIN's ITR<0 path computes the ground reflection loss from the Fresnel
coefficients of SIGPAT/EPSPAT.  It is dense in transcendentals (COS, SIN, ATAN,
ASIN, SQRT, ALOG) and the CV/CH numerators subtract comparable terms, so it
carries the [[f32-libm-noise]] floor with some cancellation amplification.

The 374 cases span sea water through ice across 2-30 MHz and all elevations,
plus the branch edges: SIGMA<=0 (no ground data -> 0) and DELTA at/below the
1e-8 grazing floor (-> a fixed 6 dB).  EFF is 0 on this path.

Measured: 98.4% bit-exact, worst 2.0 floored ULP -- tight for six chained
transcendentals, so the cancellation in CV/CH turns out to be mild here.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.gain import ground_loss
from voacap_gpu.scalar.layerstate import LayerState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("gain")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_gain_dumps.sh")
def test_gain_ground_loss_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["rain", "eff"]
    total = exact = 0
    worst = 0.0
    worst_where = ""
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        ig = int(inp[0])
        sigma, er, delta, fmc = (float(x) for x in inp[1:5])
        s = LayerState()
        s.sigpat[ig] = F32(sigma)
        s.epspat[ig] = F32(er)
        got = ground_loss(s, ig, delta, fmc)
        e = np.float32(exp[0])
        d = ulp_diff_floored(got, e)
        if d > worst:
            worst, worst_where = d, f"case {k}: {got!r} vs {e!r} in {inp}"
        assert d <= 8, (f"case {k} rain: got {got!r} expected {e!r} "
                        f"({d:.2f} floored ULP) in {inp}")
        total += 1
        exact += d == 0.0
        # EFF is untouched on the ground path
        assert np.float32(exp[1]) == np.float32(0.0), f"case {k} eff not 0"
    assert exact / total >= 0.95, (f"exact {exact}/{total} "
                                   f"(worst {worst:.2f} @ {worst_where})")
